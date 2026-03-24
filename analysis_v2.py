
import json
from collections import Counter, defaultdict
from types import SimpleNamespace
from pathlib import Path
import pickle
import time
# Math
import math
import numpy as np
import pandas as pd
from numba import njit, float64, types
from numba.experimental import jitclass
from numba.typed import Dict
from scipy.signal import savgol_filter
# Plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
from matplotlib.ticker import ScalarFormatter, FuncFormatter
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

from particle import Particle
import vector
import jetnet
import jetnet.evaluation
# particleGPT
import pUtil
from dictionary import Dictionary
import data_manager

class dataset:
    
    @staticmethod
    def get_pdgid_frequency_distribution(dataset):
        """
        dataset: expected shape (num_events, num_particles_per_event, num_features_per_particle)
            where the first feature is the PDGID of the particle.
        freq: contains the frequency of each PDGID in the dataset.
        occurrences: contains the events (as row numbers) in which each PDGID occurs.
        """
        freq = Counter()
        occurrences = defaultdict(list)
        
        for event_idx, event in enumerate(dataset):
            found_ids = set()
            for particle in event:
                pdgid = particle[0]
                if pdgid == 0.0 or pdgid == np.nan:
                    continue
                freq[pdgid] += 1
                found_ids.add(pdgid)
            for pid in found_ids:
                occurrences[pid].append(event_idx)
                
        return freq, occurrences
    
    @staticmethod
    def find_events_lost_due_to_particle_removal(rows_occurring, n_least_frequent):
        """
        events_lost: If we remove all events containing the n_least_frequent particles, which events (as row numbers) do we lose?
        """
        sorted_items = sorted(rows_occurring.items(), key=lambda x: len(x[1]), reverse=False)
        sorted_items = sorted_items[:n_least_frequent]
        
        events_lost = set()
        for particle, events in sorted_items:
            events_lost.update(events)
        return events_lost

    @staticmethod
    def calculate_num_removable_particles(rows_occurring, n_allowed_event_removals):
        """
        n_most_remove_particles: number of least frequent particles we can remove and still only remove n_allowed_event_removals events.
        """
        n_most_removable_particles = 0
        for i in range(0, len(rows_occurring)):
            removed_events = dataset.find_events_lost_due_to_particle_removal(rows_occurring, i)
            n_removed_events = len(removed_events)
            if n_removed_events <= n_allowed_event_removals:
                n_most_removable_particles = i
        return n_most_removable_particles

    # ===================== 
    # Process data for various analysis 
    # =====================

    @staticmethod
    def extract_single_column_for_analysis(in_dataset, column_name, return_only_leading=False):
        """
        Processes in_dataset for analysis. This includes
        1) Removing incident particle from dataset (i.e. only keeping outgoing particles)
        2) Removing padding events (IMPORTANT: padding particles must be np.nan)
        3) Extracting the column of interest (e.g. 'pt', 'eta', etc.)
        4) Optionally, returning only the leading particle (i.e. the particle with the highest energy) for each event.
        
        @return
        processed_data: np.ndarray of shape (n_events, n_particles_per_event) or (n_events,) if return_only_leading=True
        """
        # @TODO: add verification that in_dataset is valid
        
        if column_name == 'num_particles':
            num_secondaries_across_events = []
            for event in in_dataset:
                # event has shape (num_particles (including padded), num features per particle)
                # Remove first particle (incident particle) and extract only pdgids since we only need one feature to test particle validity
                event_pdgids = event[1:, 0]
                valid_pdgids = event_pdgids[~np.isnan(event_pdgids)]
                num_secondaires = len(valid_pdgids)
                num_secondaries_across_events.append(num_secondaires)
            return np.asarray(num_secondaries_across_events)
        
        relevant_column_pos = plotting_v2.verbose_columns.index(column_name)
        # Remove incident particle
        secondaries = in_dataset[:, 1:, :]
        
        if return_only_leading:
            # Extract energies for comparison
            energies = secondaries[:, :, 1]
            # Only consider events with at least one particle
            valid_events = ~np.all(np.isnan(energies), axis=1)
            leading_idx = np.nanargmax(energies[valid_events], axis=1)
            leadings = secondaries[valid_events, :, :][np.arange(len(leading_idx)), leading_idx]
            leadings_column = leadings[:, relevant_column_pos].ravel()
            return leadings_column
        
        # Keep only column of interest
        # Flatten the array to make it easier to remove padding events
        secondaries_column = secondaries[:, :, relevant_column_pos].ravel()
        # Remove padding events (i.e. np.nan)
        secondaries_column = secondaries_column[~np.isnan(secondaries_column)]
        return secondaries_column
            
    @classmethod
    def extract_ein_eout_for_analysis(cls, model_name, in_dataset):
        MASS_CARBON = 931.5 * 12  # [MeV]
        
        e_in = np.full(len(in_dataset), np.nan, dtype=float)
        e_out = np.full(len(in_dataset), np.nan, dtype=float)

        # ===== Build PDGID mass dictionary =====
        
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / "dictionary.json"
        dictionary = Dictionary(dictionary_filename)

        pdgid_mass_dict = {}
        for _, pdgid in dictionary.pdgids.items():
            if pdgid == 0:
                continue
            part = Particle.from_pdgid(pdgid)
            pdgid_mass_dict[pdgid] = part.mass
            
        # ===== Calculate incoming and outgoing energies =====
        
        for idx, event in enumerate(in_dataset):
            # Incoming system
            in_pdgid = event[0][0]
            in_px    = event[0][2]
            in_py    = event[0][3]
            in_pz    = event[0][4]
            
            if np.isnan(in_pdgid):
                continue
            
            in_particle_vec = vector.obj(mass=pdgid_mass_dict[in_pdgid], px=in_px, py=in_py, pz=in_pz )
            in_material_vec = vector.obj(mass=MASS_CARBON, px=0.0, py=0.0, pz=0.0)
            in_vec = in_particle_vec + in_material_vec
            
            # Outgoing system
            out_vec = vector.obj(mass=0.0, px=0.0, py=0.0, pz=0.0)
            for particle in event[1:]:
                i_pdgid = particle[0]
                i_px    = particle[2]
                i_py    = particle[3]
                i_pz    = particle[4]
                
                if np.isnan(i_pdgid):
                    continue
                
                i_vec = vector.obj(mass=pdgid_mass_dict[i_pdgid], px=i_px, py=i_py, pz=i_pz)
                out_vec += i_vec
            
            e_in[idx] = in_vec.e
            e_out[idx] = out_vec.e
            
        return e_in, e_out

class metrics:
    # Wrappers for JetNet metrics
    jetnet_get_suggested_kpd_fpd_features = jetnet.evaluation.get_fpd_kpd_jet_features
    jetnet_eval_cov_mmd = jetnet.evaluation.cov_mmd
    jetnet_eval_kpd = jetnet.evaluation.kpd
    jetnet_eval_fpd = jetnet.evaluation.fpd
    jetnet_eval_w1efp = jetnet.evaluation.w1efp
    jetnet_eval_w1m = jetnet.evaluation.w1m
    jetnet_eval_w1p = jetnet.evaluation.w1p
    
    # Custom metrics implementations go here...

class plotting_v2:
    
    legend_fontsize = 12
    annotation_fontsize = 10
    axes_fontsize = 10
    ticks_fontsize = 10
    
    verbose_columns = ["pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    
    @staticmethod
    def set_publication_style(*, context="paper", font="serif", base_fontsize=10, dpi=200, save_dpi=600, use_tex=False): 
        """
        Sets seaborn theme properties for beautiful plots.
        
        @note: set use_tex=True if a valid LaTeX install is available
        """
        # Seaborn baseline
        sns.set_theme(context=context, style="ticks")

        # Colorblind-friendly palette (chatGPT says this is good for journal plots)
        sns.set_palette("colorblind")

        # Matplotlib rcParams tuned for print
        mpl.rcParams.update({
            # Figure + save
            "figure.dpi": dpi,
            "savefig.dpi": save_dpi,
            "savefig.bbox": "tight",
            "savefig.transparent": True,   # keeps backgrounds clean in PDFs
            "pdf.fonttype": 42,            # embed TrueType fonts
            "ps.fonttype": 42,

            # Typography
            "font.family": font,
            "font.size": base_fontsize,
            "axes.titlesize": base_fontsize + 1,
            "axes.labelsize": base_fontsize + 1,
            "xtick.labelsize": base_fontsize,
            "ytick.labelsize": base_fontsize,
            "legend.fontsize": base_fontsize,
            "legend.title_fontsize": base_fontsize,

            # Axes + lines
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.8,
            "lines.markersize": 5,

            # Ticks (major + minor)
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 5,
            "ytick.major.size": 5,
            "xtick.minor.size": 3,
            "ytick.minor.size": 3,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.8,
            "ytick.minor.width": 0.8,

            # Grid (usually off for journal plots; enable per-figure if needed)
            "axes.grid": False,

            # Legend
            "legend.frameon": False,

            # Math text
            "mathtext.fontset": "stix",
            "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
            "text.usetex": use_tex,
        })

        # Remove top/right spines for clean physics look
        sns.despine(trim=True)

    # Custom formatter to force scientific notation for small numbers
    def _sci_notation(x, pos):
        return f'{x:.0e}'
    
    # =====================
    # Plotting utils - Data Processing
    # =====================
    
    @staticmethod
    def _get_data_frequency_distribution(in_real_data, in_generated_data, top_k=None, sort_descending=False):
        unique_real, counts_real = np.unique(in_real_data, return_counts=True)
        gen_unique_counts_dict = {
            model_name: np.unique(vals, return_counts=True)
            for model_name, vals in in_generated_data.items()
        }
        # Create a combined set of unique particle types
        all_unique_particles = np.union1d(
            unique_real,
            np.concatenate([u for u, _ in gen_unique_counts_dict.values()])
        )
        
        # Real frequency distribution
        real_freq_dist = np.zeros(len(all_unique_particles), dtype=int)
        real_idx = np.searchsorted(all_unique_particles, unique_real)
        real_freq_dist[real_idx] = counts_real
        # Generated frequency distributions
        sampled_freq_dist_dict = {}
        for model_name, (unique_sampled, counts_sampled) in gen_unique_counts_dict.items():
            freq = np.zeros(len(all_unique_particles), dtype=int)
            idx = np.searchsorted(all_unique_particles, unique_sampled)
            freq[idx] = counts_sampled
            sampled_freq_dist_dict[model_name] = freq
            
        # Sort particles by real frequency (descending)
        if sort_descending:
            order = np.argsort(real_freq_dist)[::-1]
            all_unique_particles = all_unique_particles[order]
            real_freq_dist = real_freq_dist[order]
            for model_name in sampled_freq_dist_dict:
                sampled_freq_dist_dict[model_name] = sampled_freq_dist_dict[model_name][order]
            
        # Only consider top_k particles if top_k is specified
        if top_k is not None:
            assert top_k > 0 and top_k <= len(all_unique_particles), "top_k must be a positive integer less than or equal to the number of unique particles."
            all_unique_particles = all_unique_particles[:top_k]
            real_freq_dist = real_freq_dist[:top_k]
            for model_name in sampled_freq_dist_dict:
                sampled_freq_dist_dict[model_name] = sampled_freq_dist_dict[model_name][:top_k]
                
        return all_unique_particles, real_freq_dist, sampled_freq_dist_dict

    # =====================
    # Plotting utils - Smoothing Functions
    # =====================
    
    @staticmethod
    def _ema(y, alpha=0.02):
        y = np.asarray(y, dtype=float)
        out = np.empty_like(y)
        out[0] = y[0]
        for i in range(1, len(y)):
            out[i] = alpha*y[i] + (1-alpha)*out[i-1]
        return out
    
    @staticmethod
    def _smooth_rolling(y, window=101, method="median_then_mean"):
        """
        window should be odd. Try 101–401 depending on noise.
        """
        s = pd.Series(np.asarray(y, dtype=float))
        if method == "median_then_mean":
            y1 = s.rolling(window, center=True, min_periods=1).median()
            y2 = y1.rolling(window, center=True, min_periods=1).mean()
            return y2.to_numpy()
        elif method == "median":
            return s.rolling(window, center=True, min_periods=1).median().to_numpy()
        elif method == "mean":
            return s.rolling(window, center=True, min_periods=1).mean().to_numpy()
        else:
            raise ValueError("Unknown method")
            
    @staticmethod
    def _smooth_savgol(y, window=301, poly=3):
        window = int(window)
        if window % 2 == 0:
            window += 1
        y = np.asarray(y, dtype=float)
        # if too short, fallback
        if len(y) < window:
            return y
        return savgol_filter(y, window_length=window, polyorder=poly, mode="interp")
    
    @staticmethod
    def _smooth_bin_average(x, y, bin_size=25):
        x = np.asarray(x)
        y = np.asarray(y, dtype=float)
        n = len(y)
        m = n // bin_size
        x2 = x[:m*bin_size].reshape(m, bin_size).mean(axis=1)
        y2 = y[:m*bin_size].reshape(m, bin_size).mean(axis=1)
        return x2, y2

    # =====================
    # Plotting utils - Aesthetics
    # =====================
    
    @staticmethod
    def _get_labels(column_name):
        # assert column_name in plotting.verbose_columns, f"Invalid column name: {column_name}. Must be one of {plotting.verbose_columns}."
        
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '[MeV]'
                        
        feature_name = column_name
        if column_name == 'pt':
            feature_name = r'$p_{T}$'
        elif column_name == 'px':
            feature_name = r'$p_{x}$'
        elif column_name == 'py':
            feature_name = r'$p_{y}$'
        elif column_name == 'pz':
            feature_name = r'$p_{z}$'
        elif column_name == 'eta':
            feature_name = r'$\eta$'
        elif column_name == 'theta':
            feature_name = r'$\theta$'
        elif column_name == 'phi':
            feature_name = r'$\phi$'
        elif column_name == 'num_particles':
            feature_name = 'Outgoing Particles Count'
            return feature_name, unit
        elif column_name == 'pdgid':
            feature_name = 'PDGID'
            return feature_name, unit
        elif column_name == 'energy_conservation':
            feature_name = r"$ \Delta E $"
        return feature_name, unit

    @staticmethod
    def _apply_dynamic_count_scaling_ticks(ax, y_arrays, base_label="Counts", group_by_3=False):
        """
        Dynamically scaled y-axis for counts plots. If the maximum count is large, the y-axis will
            be scaled down by a power of 10 and the label will indicate this.
        e.g. instead of 100,000, 200,000, etc. the y-axis will show 1, 2, etc. and the label will say "Counts (×10^5)".
        """
        y_all = np.concatenate([np.asarray(y) for y in y_arrays if y is not None])
        y_max = np.nanmax(y_all) if y_all.size else 0
        if not np.isfinite(y_max) or y_max <= 0:
            ax.set_ylabel(base_label)
            return

        order = int(np.floor(np.log10(y_max)))
        if group_by_3:
            order = int(np.floor(order / 3) * 3)
        scale = 10 ** order

        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/scale:g}"))
        # ax.set_ylabel(f"{base_label} (×10$^{{{order}}}$)")
        ax.set_ylabel(rf"Counts ($\times 10^{{{order}}}$)")
    
    @staticmethod
    def _place_legend_emptiest_corner(ax, x, y, labels, colors=None, lw=2, fontsize=legend_fontsize):
        """
        Choose legend location among 4 corners by finding the corner with minimal summed y
        in that x-range and near-top y-range. Works well for peaked distributions. Avoids issues
        with legend overlapping data.
        """
        x = np.asarray(x)
        y = np.asarray(y)

        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Define corner x-windows as fractions of axis span
        xL0, xL1 = xmin + 0.00*(xmax-xmin), xmin + 0.35*(xmax-xmin)
        xR0, xR1 = xmin + 0.65*(xmax-xmin), xmin + 1.00*(xmax-xmin)

        # Score corners by "how much curve lives there"
        def score(x0, x1):
            m = (x >= x0) & (x <= x1)
            if not np.any(m):
                return np.inf
            return np.nansum(y[m])

        scores = {
            "upper left":  score(xL0, xL1),
            "upper right": score(xR0, xR1),
            "lower left":  score(xL0, xL1) + 1e9,   # discourage lower corners for hists usually
            "lower right": score(xR0, xR1) + 1e9,
        }

        best_loc = min(scores, key=scores.get)
        
        # This is here to move the left corner slightly rightwards to avoid clashing with the panel labels
        if best_loc == "upper left":
            leg = ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.98), frameon=False, fontsize=fontsize)
        else:
            leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=fontsize)

        ax.add_artist(leg)
        return leg
    
    @classmethod
    def _place_annotation_emptiest_corner(cls, fig, axd, x, y, labels, leg_bbox_pixels=None, spec_text=None, spec_table_rows=None, fontsize=legend_fontsize):
        """
        Choose annotation location among 4 corners by finding the corner with minimal summed y
        in that x-range and near-top y-range. Works well for peaked distributions. Further, avoid
        overlapping with the legend. Avoids issues with annotation overlapping data.
        """
        # x = np.asarray(x)
        # y = np.asarray(y)

        # xmin, xmax = ax.get_xlim()
        # ymin, ymax = ax.get_ylim()

        # # Define corner x-windows as fractions of axis span
        # xL0, xL1 = xmin + 0.00*(xmax-xmin), xmin + 0.35*(xmax-xmin)
        # xR0, xR1 = xmin + 0.65*(xmax-xmin), xmin + 1.00*(xmax-xmin)

        # # Score corners by "how much curve lives there"
        # def score(x0, x1):
        #     m = (x >= x0) & (x <= x1)
        #     if not np.any(m):
        #         return np.inf
        #     return np.nansum(y[m])

        # scores = {
        #     "upper left":  score(xL0, xL1),
        #     "upper right": score(xR0, xR1),
        #     "lower left":  score(xL0, xL1) + 1e9,   # discourage lower corners for hists usually
        #     "lower right": score(xR0, xR1) + 1e9,
        # }

        # # best is for legend, second best is for annotation
        # second_best_loc = sorted(scores, key=scores.get)[1]
        
        # # Make a fake legend to get geometry at the second best location. Remove it after.
        # if second_best_loc == "upper left":
        #     temp_leg = ax.legend(loc="upper left", bbox_to_anchor=(0.03, 0.98), frameon=False, fontsize=fontsize)
        # else:
        #     temp_leg = ax.legend(loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False, fontsize=fontsize)
        # fig.canvas.draw()
        # bbox_leg_pixels = temp_leg.get_window_extent()
        # bbox_leg_axes = bbox_leg_pixels.transformed(ax.transAxes.inverted())
        # temp_leg.remove()
        
        
        
        if spec_text is not None:
            leg_bbox_axes = leg_bbox_pixels.transformed(axd.transAxes.inverted())
            
            # Get the width of the annotation so we can do better placement calculations. Remove right after.
            temp_annotation = axd.text(
                0, 0, s=spec_text, transform=axd.transAxes,
                fontsize=cls.annotation_fontsize, va="top", ha="left", linespacing=1.25,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#999999",
                    linewidth=2,
                    boxstyle="round,pad=0.4"
                )
            )
            fig.canvas.draw()
            bbox_annotation_pixels = temp_annotation.get_window_extent()
            bbox_annotation_axes = bbox_annotation_pixels.transformed(axd.transAxes.inverted())
            temp_annotation.remove()
            
            # Find if legend is right or left within the figure and adjust annotation placement accordingly
            fig_width_pixels = fig.get_window_extent().width
            if leg_bbox_pixels.x0 > fig_width_pixels / 2:
                delta_width = (leg_bbox_axes.width * 0.95) - bbox_annotation_axes.width
                annotation_loc = (leg_bbox_axes.x0 + delta_width, leg_bbox_axes.y0 - 0.02)
            else:
                annotation_loc = (leg_bbox_axes.x0, leg_bbox_axes.y0 - 0.02)
                        
            axd.text(
                *annotation_loc, s=spec_text, transform=axd.transAxes,
                fontsize=cls.annotation_fontsize, va="top", ha="left", linespacing=1.25,
                bbox=dict(
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#999999",
                    linewidth=2,
                    boxstyle="round,pad=0.4"
                )
            )
        elif spec_table_rows is not None:
            fig.canvas.draw() # legend must be rendered before we use location properties
            leg_bbox_axes = leg_bbox_pixels.transformed(axd.transAxes.inverted())
            
            annotation_width = 0.26
            
            # Find if legend is right or left within the figure and adjust annotation placement accordingly
            fig_width_pixels = fig.get_window_extent().width
            if leg_bbox_pixels.x0 > fig_width_pixels / 2:
                delta_width = (leg_bbox_axes.width * 0.95) - annotation_width
            else:
                delta_width = 0
            
            x0 = leg_bbox_axes.x0 + delta_width
            y0 = leg_bbox_axes.y0 - 0.04
            dy = 0.04
            
            col_codebook = x0 + 0.13
            col_events   = x0 + 0.2
            
            box = FancyBboxPatch(
                (x0 + 0.005, y0 - 4*dy + 0.026),
                width=annotation_width,
                height=4*dy - 0.01,
                transform=axd.transAxes,
                boxstyle="round,pad=0.02",
                facecolor="white",
                alpha=0.85,
                edgecolor="#bbbbbb",
                linewidth=2,
                zorder=2
            )

            axd.add_patch(box)
            axd.text(x0, y0, "Generator", transform=axd.transAxes, fontsize=cls.annotation_fontsize, weight="bold", zorder=3)
            axd.text(col_codebook, y0, "Size", transform=axd.transAxes, fontsize=cls.annotation_fontsize, weight="bold", zorder=3)
            axd.text(col_events, y0, "N", transform=axd.transAxes, fontsize=cls.annotation_fontsize, weight="bold", zorder=3)
        
            rows = spec_table_rows
            for i, (g, c, e) in enumerate(rows):
                y = y0 - (i + 1) * dy

                axd.text(x0, y, g, transform=axd.transAxes, fontsize=cls.annotation_fontsize, zorder=3)
                axd.text(col_codebook, y, c, transform=axd.transAxes, fontsize=cls.annotation_fontsize, zorder=3)
                axd.text(col_events, y, e, transform=axd.transAxes, fontsize=cls.annotation_fontsize, zorder=3)
        
        # return bbox_leg_axes

    @staticmethod
    def get_common_data(model_name ):
        """
        Retrieves bin widths and ranges for each feature and the real and sampled leading particles dataframes.
        """
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
        dictionary = Dictionary(dictionary_filename)
        
        def get_bin_count(type_str):
            step_size = dictionary.token_step_size(type_str)
            if type_str in ['eta', 'theta', 'phi']:
                step_size = 0.1
            if type_str == 'theta':
                theta_min = 0 if dictionary.token_min('theta') == 0 else dictionary.token_min('theta')
                theta_max = np.pi if dictionary.token_max('theta') == 0 else dictionary.token_max('theta')
                return int((theta_max - theta_min) // step_size)
            if type_str == 'phi':
                phi_min = -np.pi if dictionary.token_min('phi') == 0 else dictionary.token_min('phi')
                phi_max = np.pi if dictionary.token_max('phi') == 0 else dictionary.token_max('phi')
                return int((phi_max - phi_min) // step_size)
            return int(dictionary.token_range(type_str) // step_size)
        
        # For now, I have replaced those with hard coded values.
        theta_min = 0 if dictionary.token_min('theta') == 0 else dictionary.token_min('theta')
        theta_max = np.pi if dictionary.token_max('theta') == 0 else dictionary.token_max('theta')
        phi_min = -np.pi if dictionary.token_min('phi') == 0 else dictionary.token_min('phi')
        phi_max = np.pi if dictionary.token_max('phi') == 0 else dictionary.token_max('phi')
        
        bin_settings = {
            "num_particles":       { "min": -0.5,                          "max": 50.5,                          "bins": 51 },
            "energy_conservation": { "min": 0,                             "max": 45000,                         "bins": 41 },
            "e":                   { "min": 0,                             "max": 35000,                         "bins": 70 },
            "px":                  { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "py":                  { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "pz":                  { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "eta":                 { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": get_bin_count('eta') },
            "theta":               { "min": theta_min,                     "max": theta_max,                     "bins": get_bin_count('theta') },
            "phi":                 { "min": phi_min,                       "max": phi_max,                       "bins": get_bin_count('phi') },
            "pt":                  { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": 70 },
        }

        return bin_settings

    # =====================
    # Training/Validation Run Plotting
    # =====================

    @classmethod
    def plot_training_run(
        cls,
        model_names,
        model_legend_title,
        show_train_loss=False,
        show_val_loss=False,
        show_best_marker=False,
        y_lim=None,
        x_lim=None,
        use_log=False,
        out_file=None,
        show_lr_schedule=False,
        figsize=(6.9, 3.2),
        show_raw=True,
        smooth_alpha=0.02,
        mark_best=True,
        smoothing_function="bin_average", # options: "bin_average", "savgol", "rolling"
    ):
        palette = sns.color_palette("colorblind", n_colors=max(3, len(model_names) + 1))
        
        model_names = np.atleast_1d(model_names)

        fig, ax = plt.subplots(figsize=figsize)

        ax.set_xlabel("Iteration")
        y_label_loss_type = ""
        if show_train_loss and not show_val_loss:
            y_label_loss_type = "Training"
        if show_val_loss and not show_train_loss:
            y_label_loss_type = "Validation"
        if show_val_loss and show_train_loss:
            y_label_loss_type = ""
            
        ax.set_ylabel(f"{y_label_loss_type} Loss")

        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if use_log:
            ax.set_yscale("log")

        ax.minorticks_on()
        ax.tick_params(which="both", direction="in", top=True, right=True)

        # very light grid
        ax.grid(True, which="major", alpha=0.25, linewidth=0.8)
        ax.grid(True, which="minor", alpha=0.12, linewidth=0.6)
        
        # Construct labels as we plot everything to ensure everything has one label of the correct attributes
        unique_handles = {}

        lrax = None
        for idx, model_name in enumerate(model_names):
            model_data = tables.get_all_data(model_name)
            df = model_data.running_df
            x = np.asarray(df["iter"])
            yt = np.asarray(df["train_loss"])
            yv = np.asarray(df["val_loss"])

            color = palette[idx]

            # Optional LR on twin axis (keep subtle)
            if show_lr_schedule:
                config_filepath = pUtil.get_model_config_filepath(model_name)
                with open(config_filepath, "r") as f:
                    model_config = json.load(f)
                    training_conf = model_config.get("training_config", {})
                    lr_scheduler = training_conf.get("lr_scheduler", "cosine_annealing_with_warmup")
                    warmup_iters = training_conf.get("warmup_iters", 0)
                    lr_decay_iters = training_conf.get("lr_decay_iters", 0)
                    learning_rate = training_conf.get("learning_rate", 0)
                    min_lr = training_conf.get("min_lr", 0)
                    base_lr_decay_mult = training_conf.get("base_lr_decay_mult", 1)
                    cycle_steps_mult = training_conf.get("cycle_steps_mult", 1)

                iters = np.arange(df["iter"].max())
                lrs = [
                    plotting_old3._get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters,
                                        min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult)
                    for it in iters
                ]

                if lrax is None:
                    lrax = ax.twinx()
                    lrax.set_ylabel("Learning rate")
                    lrax.tick_params(which="both", direction="in", top=True, right=True)
                    lrax.grid(False)
                    lrax.yaxis.set_major_formatter(FuncFormatter(plotting_v2._sci_notation))

                # make LR faint so it doesn't dominate
                lrax.plot(iters, lrs, color='black', linewidth=1.0, linestyle=':', alpha=0.35, label="LR")
                
                # Append to handles
                unique_handles['Learning Rate'] = mpl.lines.Line2D([0], [0], color='black', linewidth=1.0, linestyle=":", alpha=0.35, label="Learning Rate")

            if show_train_loss == True:
                if show_raw:
                    ax.plot(x, yt, linewidth=1.0, alpha=0.25, color=color)
                
                if smoothing_function == "savgol":
                    yt_s = plotting_v2._smooth_savgol(yt, window=301, poly=3)
                    ax.plot(x, yt_s, linewidth=2.2, color=color, label=model_legend_title[idx], linestyle='--')
                elif smoothing_function == "rolling":
                    yt_s = plotting_v2._smooth_rolling(yt, window=201, method="median_then_mean") 
                    ax.plot(x, yt_s, linewidth=2.2, color=color, label=model_legend_title[idx], linestyle='--')
                elif smoothing_function == "bin_average":
                    xt_s, yt_s = plotting_v2._smooth_bin_average(x, yt, bin_size=12)
                    ax.plot(xt_s, yt_s, linewidth=2.2, color=color, label=model_legend_title[idx], linestyle='--')
                    
                # Mark best checkpointed train loss
                if show_best_marker and mark_best and hasattr(model_data, "checkpointed_df") and len(model_data.checkpointed_df) > 0:
                    best = model_data.checkpointed_df.loc[model_data.checkpointed_df["train_loss"].idxmin()]
                    ax.scatter([best["iter"]], [best["train_loss"]], s=22, color=color, zorder=5, marker='o')
                
                # Append to handles
                linestype = '-' if show_val_loss else '--'
                unique_handles[model_legend_title[idx]] = mpl.lines.Line2D([0], [0], linestyle=linestype, color=color, linewidth=2, label=model_legend_title[idx])
                
            if show_val_loss == True:
                if show_raw:
                    ax.plot(x, yv, linewidth=1.0, alpha=0.25, color=color)
                
                if smoothing_function == "savgol":
                    yv_s = plotting_v2._smooth_savgol(yv, window=301, poly=3)
                    ax.plot(x, yv_s, linewidth=2.2, color=color, label=model_legend_title[idx])
                elif smoothing_function == "rolling":
                    yv_s = plotting_v2._smooth_rolling(yv, window=201, method="median_then_mean") 
                    ax.plot(x, yv_s, linewidth=2.2, color=color, label=model_legend_title[idx])
                elif smoothing_function == "bin_average":
                    xv_s, yv_s = plotting_v2._smooth_bin_average(x, yv, bin_size=12)
                    ax.plot(xv_s, yv_s, linewidth=2.2, color=color, label=model_legend_title[idx])

                # Mark best checkpointed val loss
                if show_best_marker and mark_best and hasattr(model_data, "checkpointed_df") and len(model_data.checkpointed_df) > 0:
                    best = model_data.checkpointed_df.loc[model_data.checkpointed_df["val_loss"].idxmin()]
                    ax.scatter([best["iter"]], [best["val_loss"]], s=22, color=color, zorder=5, marker='s')
                
                # Append to handles
                unique_handles[model_legend_title[idx]] = mpl.lines.Line2D([0], [0], linestyle='-', color=color, linewidth=2, label=model_legend_title[idx])

        unique_handles = list(unique_handles.values())
        ax.legend(handles=unique_handles, loc="best", frameon=False)
        fig.subplots_adjust(right=0.80)

        sns.despine(ax=ax, trim=True)
        fig.tight_layout()

        if out_file is not None:
            fig.savefig(out_file, bbox_inches="tight")

        plt.show()
        return fig, ax

    # =====================
    # Distribution Plotting
    # =====================

    @classmethod
    def plot_dist_and_ratio_cont(
        cls,
        *,
        column_name,
        ref_vals,
        comp_vals_dict,             # dict: model_name -> np.array
        model_legend_titles,        # ["Geant4", "model1 label", "model2 label", ...] aligned with model_names
        edges=None,
        density=False,
        use_log=False,
        out_file=None,
        title=None,
        ratio_ylim=(0.5, 1.5),
        ratio_bands=(0.05, 0.10, 0.20),
        dynamic_count_scale=True,
        count_scale_group_by_3=False,
        spec_text=None,
        spec_table_rows=None,
    ):
        # ===== Defaults if not set =====
        
        model_names = list(comp_vals_dict.keys())
        palette = sns.color_palette("colorblind", n_colors=max(3, len(model_names) + 1))
        
        feature_name, unit = cls._get_labels(column_name)
        
        # Common bins from first model
        if edges is None:
            bin_settings = cls.get_common_data(model_names[0])
            x_min = bin_settings[column_name]["min"]
            x_max = bin_settings[column_name]["max"]
            n_bins = bin_settings[column_name]["bins"]
            edges = np.linspace(x_min, x_max, n_bins + 1)

        centers = 0.5 * (edges[:-1] + edges[1:])
        widths = np.diff(edges)

        fig, (axd, axr) = plt.subplots(
            2, 1,
            figsize=(6, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )

        # ===== Writing =====

        # Useful for slides etc, where the figure might be standalone
        if title:
            fig.suptitle(title, fontsize=12)

        # Labels
        if density:
            axd.set_ylabel("Density")
        else:
            axd.set_ylabel("log(Counts)" if use_log else "Counts")

        axr.set_ylabel("Gen / Geant4")
        axr.set_xlabel(f"{feature_name} {unit}".rstrip())

        # ===== Axes =====

        axd.set_yscale("log" if use_log else "linear")

        for ax in (axd, axr):
            ax.minorticks_on()
            ax.tick_params(which="both", direction="in", top=True, right=True)
        
       #  axd.yaxis.set_major_locator(mticker.MaxNLocator(6))
       #  axr.yaxis.set_major_locator(mticker.MultipleLocator(0.1))
        
        # ===== Reference Model =====

        ref_vals = np.asarray(ref_vals)
        rcounts, _ = np.histogram(ref_vals, bins=edges)
        rN = np.sum(rcounts)
        
        if density:
            r_y = rcounts / (rN * widths) if rN > 0 else np.zeros_like(rcounts, dtype=float)
            r_err = np.sqrt(rcounts) / (rN * widths) if rN > 0 else np.zeros_like(rcounts, dtype=float)
        else:
            r_y = rcounts.astype(float)
            r_err = np.sqrt(rcounts)

        axd.stairs(
            r_y, edges, color="black",
            linewidth=2.2, label=f"{model_legend_titles[0]}", zorder=4, alpha=0.5
        )

        mask_r = rcounts > 0
        axd.errorbar(
            centers[mask_r], r_y[mask_r], yerr=r_err[mask_r],
            fmt="none", ecolor="black", elinewidth=0.9, capsize=0, alpha=0.9, zorder=5
        )

        # Ratio denominator (densities, shape-only)
        r_density = rcounts / rN if rN > 0 else np.zeros_like(rcounts, dtype=float)
        r_density_err = np.sqrt(rcounts) / rN if rN > 0 else np.zeros_like(rcounts, dtype=float)

        top_y_arrays = [r_y]

        # ===== Comparison Models =====
        
        # Make a dict of models -> num events for optional annotations later
        model_to_num_events = {}
        
        for i, mn in enumerate(model_names):
            gvals = np.asarray(comp_vals_dict[mn])
            gcounts, _ = np.histogram(gvals, bins=edges)
            gN = np.sum(gcounts)
            # Store for later use (e.g. annotations)
            model_to_num_events[mn] = gN

            if density:
                g_y = gcounts / (gN * widths) if gN > 0 else np.zeros_like(gcounts, dtype=float)
                g_err = np.sqrt(gcounts) / (gN * widths) if gN > 0 else np.zeros_like(gcounts, dtype=float)
            else:
                g_y = gcounts.astype(float)
                g_err = np.sqrt(gcounts)

            top_y_arrays.append(g_y)

            color = palette[i + 1]
            axd.stairs(
                g_y, edges, linewidth=2.0, color=color,
                label=f"{model_legend_titles[i+1]}", zorder=2
            )

            mask_g = gcounts > 0
            axd.errorbar(
                centers[mask_g], g_y[mask_g], yerr=g_err[mask_g],
                fmt="none", ecolor=color, elinewidth=0.8, capsize=0, alpha=0.9, zorder=3
            )

            # Ratio (densities)
            g_density = gcounts / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)
            g_density_err = np.sqrt(gcounts) / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)

            ratio = np.divide(g_density, r_density, out=np.full_like(g_density, np.nan, dtype=float), where=r_density > 0)

            # Error propagation on ratio
            ratio_err = np.full_like(ratio, np.nan, dtype=float)
            ok = (g_density > 0) & (r_density > 0)
            ratio_err[ok] = ratio[ok] * np.sqrt(
                (g_density_err[ok] / g_density[ok])**2 + (r_density_err[ok] / r_density[ok])**2
            )

            axr.stairs(ratio, edges, linewidth=1.6, color=color, zorder=2)
            ok2 = np.isfinite(ratio) & np.isfinite(ratio_err)
            axr.errorbar(
                centers[ok2], ratio[ok2], yerr=ratio_err[ok2],
                fmt="none", ecolor=color, elinewidth=0.8, capsize=0, alpha=0.9, zorder=3
            )
            
        # ===== Aesthetics =====
            
        axd.text(0.015, 0.95, "(a)", transform=axd.transAxes, fontsize=11, va="top")
        axr.text(0.015, 0.95, "(b)", transform=axr.transAxes, fontsize=11, va="top")

        # Ratio guide lines
        axr.axhline(1.0, color="0.25", lw=1.2)
        for frac in [0.05, 0.10]:
            axr.axhline(1.0 + frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
            axr.axhline(1.0 - frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
        axr.set_ylim(*ratio_ylim)
        
        # 10% top headroom for less cramped feel
        ymin, ymax = axd.get_ylim()
        if use_log:
            log_range = np.log10(ymax) - np.log10(ymin)
            new_log_top = np.log10(ymax) + 0.10 * log_range
            axd.set_ylim(ymin, 10**new_log_top)
        else:
            axd.set_ylim(ymin, ymax * 1.10)
        
        # Fix for log scale axes tick label issue
        # axd.yaxis.set_major_locator(mticker.LogLocator(base=10))
        # axd.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))

        # Dynamic y scaling for counts (not density and not log because those do not need it)
        if dynamic_count_scale and (not density) and (not use_log):
            cls._apply_dynamic_count_scaling_ticks(axd, top_y_arrays, base_label="Counts", group_by_3=count_scale_group_by_3)

        # ===== Legend =====
        
        # Choose legend position dynamically (based on reference curve)
        x_for_score = 0.5 * (edges[:-1] + edges[1:])
        y_for_score = r_y   # use Geant4 reference to avoid peak overlap
        leg = cls._place_legend_emptiest_corner(axd, x_for_score, y_for_score, labels=None, fontsize=cls.legend_fontsize)
        
        # ===== Optional Model Spec Annotations =====

        fig.canvas.draw() # legend must be rendered before we use location properties
        leg_bbox_pixels = leg.get_window_extent()
        cls._place_annotation_emptiest_corner(
            fig, axd, x_for_score, y_for_score, labels=None, leg_bbox_pixels=leg_bbox_pixels,
            spec_text=spec_text, spec_table_rows=spec_table_rows, fontsize=cls.legend_fontsize
        )

        # ===== Finishing and show/save =====

        sns.despine(ax=axd, trim=True)
        sns.despine(ax=axr, trim=True)

        if out_file is not None:
            fig.savefig(out_file, bbox_inches="tight")
        plt.show()
        return fig, (axd, axr)

    @classmethod
    def plot_dist_and_ratio_discrete_overlaid(
        cls,
        *,
        column_name,
        ref_vals,
        comp_vals_dict,             # dict: model_name -> np.array
        model_legend_titles,        # ["Geant4", "model1 label", "model2 label", ...] aligned with model_names
        density=False,
        use_log=False,
        out_file=None,
        title=None,
        ratio_ylim=(0.5, 1.5),
        ratio_bands=(0.05, 0.10, 0.20),
        dynamic_count_scale=True,
        count_scale_group_by_3=False,
        spec_text=None,
        spec_table_rows=None,
        top_k=None,
        sort_descending=False
    ):
        """
        Plots discrete data as a bargraph where all bars are grouped together.
        """

        # ===== Defaults if not set =====
        
        model_names = list(comp_vals_dict.keys())
        palette = sns.color_palette("colorblind", n_colors=max(3, len(model_names) + 1))
        
        feature_name, unit = cls._get_labels(column_name)
        
        # ===== Get data frequency distribution =====
        
        all_unique_elements, real_freq_dist, sampled_freq_dist_dict = plotting_v2._get_data_frequency_distribution(
            in_real_data=ref_vals,
            in_generated_data=comp_vals_dict,
            top_k=top_k,
            sort_descending=sort_descending
        )
        
        # Keep raw counts for error propagation
        real_counts_raw = real_freq_dist.copy()
        sampled_counts_raw_dict = {
            mn: sampled_freq_dist_dict[mn].copy()
            for mn in model_names
        }
        
        # ===== Normalize / top-panel errors =====
        
        if density:
            real_total = real_counts_raw.sum()
            if real_total > 0:
                r_y = real_counts_raw.astype(float) / real_total
                r_err = np.sqrt(real_counts_raw.astype(float)) / real_total
            else:
                r_y = real_counts_raw.astype(float)
                r_err = np.zeros_like(r_y, dtype=float)

            g_y_dict = {}
            g_err_dict = {}
            for mn in model_names:
                g_total = sampled_counts_raw_dict[mn].sum()
                if g_total > 0:
                    g_y_dict[mn] = sampled_counts_raw_dict[mn].astype(float) / g_total
                    g_err_dict[mn] = np.sqrt(sampled_counts_raw_dict[mn].astype(float)) / g_total
                else:
                    g_y_dict[mn] = sampled_counts_raw_dict[mn].astype(float)
                    g_err_dict[mn] = np.zeros_like(g_y_dict[mn], dtype=float)
        else:
            r_y = real_counts_raw.astype(float)
            r_err = np.sqrt(real_counts_raw.astype(float))

            g_y_dict = {
                mn: sampled_counts_raw_dict[mn].astype(float)
                for mn in model_names
            }
            g_err_dict = {
                mn: np.sqrt(sampled_counts_raw_dict[mn].astype(float))
                for mn in model_names
            }
            
        # ===== Ratio denominator always uses shape-only densities =====

        rN = real_counts_raw.sum()
        r_density = real_counts_raw.astype(float) / rN if rN > 0 else np.zeros_like(real_counts_raw, dtype=float)
        # Ratio error propagation: sqrt(N) / N = 1 / sqrt(N)
        r_density_err = np.sqrt(real_counts_raw.astype(float)) / rN if rN > 0 else np.zeros_like(real_counts_raw, dtype=float)
        
        fig, (axd, axr) = plt.subplots(
            2, 1,
            figsize=(6, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
        
        # ===== Writing =====

        # Useful for slides etc, where the figure might be standalone
        if title:
            fig.suptitle(title, fontsize=12)

        # Labels
        den_or_cnts = "Density" if density else "Counts"
        axd.set_ylabel(f"log({den_or_cnts})" if use_log else den_or_cnts)
        
        axr.set_ylabel("Gen / Geant4")
        axr.set_xlabel(f"{feature_name} {unit}".rstrip())

        # ===== Axes =====

        axd.set_yscale("log" if use_log else "linear")

        for axd in ([axd]):
            axd.minorticks_on()
            axd.tick_params(which="both", direction="in", top=True, right=True)
        
        all_unique_particles_ticks = [str(int(p)) for p in all_unique_elements]
        plt.xticks(rotation=45, ha='right')
        
        # ===== X positions =====

        n_categories = len(all_unique_elements)
        x = np.arange(n_categories)

        tick_labels = [str(int(p)) for p in all_unique_elements]

        # For legend scoring / annotations
        top_y_arrays = [r_y]
    
        # ===== Top panel: grouped bars =====
        
        error_kw = dict(linewidth=1, capsize=0, capthick=0)
        
        # For log scale, error bars cannot go to/below zero
        if use_log:
            eps = 1e-12
            if density:
                real_freq_dist = np.clip(real_freq_dist, eps, None)
                for mn in model_names:
                    sampled_freq_dist_dict[mn] = np.clip(sampled_freq_dist_dict[mn], eps, None)
        
        # Do plot
        axd.bar(
            x, r_y, yerr=r_err,
            label=model_legend_titles[0], color='black', linewidth=0, alpha=0.8, width=0.9, align='center', error_kw=error_kw
        )
        for i, mn in enumerate(model_names, start=1):
            g_y = g_y_dict[mn]
            g_err = g_err_dict[mn]
            top_y_arrays.append(g_y)

            axd.bar(
                x, g_y_dict[mn], yerr=g_err,
                label=model_legend_titles[i], color=palette[i], linewidth=0, alpha=0.8, width=(0.9 - 0.3 * i), align='center', error_kw=error_kw
            )
            
        axd.set_xticks(x)
        axd.set_xticklabels(all_unique_particles_ticks, rotation=45, ha="right")
        
        # ===== Bottom panel: ratios (shape-only densities) =====

        for i, mn in enumerate(model_names, start=1):
            color = palette[i]

            gcounts = sampled_counts_raw_dict[mn].astype(float)
            gN = gcounts.sum()

            g_density = gcounts / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)
            g_density_err = np.sqrt(gcounts) / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)

            ratio = np.divide(
                g_density,
                r_density,
                out=np.full_like(g_density, np.nan, dtype=float),
                where=r_density > 0
            )

            ratio_err = np.full_like(ratio, np.nan, dtype=float)
            ok = (g_density > 0) & (r_density > 0)
            ratio_err[ok] = ratio[ok] * np.sqrt(
                (g_density_err[ok] / g_density[ok])**2 +
                (r_density_err[ok] / r_density[ok])**2
            )

            ok2 = np.isfinite(ratio) & np.isfinite(ratio_err)

            axr.errorbar(
                (x)[ok2],
                ratio[ok2],
                yerr=ratio_err[ok2],
                fmt="o",
                ms=3.5,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                alpha=0.95,
                zorder=3
            )
        
        # ===== Aesthetics =====
        
        axd.text(0.015, 0.95, "(a)", transform=axd.transAxes, fontsize=11, va="top")
        axr.text(0.015, 0.95, "(b)", transform=axr.transAxes, fontsize=11, va="top")

        # Ratio guide lines
        axr.axhline(1.0, color="0.25", lw=1.2)
        for frac in ratio_bands:
            axr.axhline(1.0 + frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
            axr.axhline(1.0 - frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
        axr.set_ylim(*ratio_ylim)

        # Top-panel headroom
        ymin, ymax = axd.get_ylim()
        if use_log:
            if ymin > 0 and ymax > 0:
                log_range = np.log10(ymax) - np.log10(ymin)
                new_log_top = np.log10(ymax) + 0.10 * log_range
                axd.set_ylim(ymin, 10**new_log_top)
        else:
            axd.set_ylim(ymin, ymax * 1.10)

        # Fix log tick formatting only when needed
        if use_log:
            axd.yaxis.set_major_locator(mticker.LogLocator(base=10))
            axd.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        
        # Dynamic y scaling for counts (not density and not log because those do not need it)
        if dynamic_count_scale and (not density) and (not use_log):
            counts_arrays = [real_freq_dist] + [sampled_freq_dist_dict[mn] for mn in model_names]
            cls._apply_dynamic_count_scaling_ticks(axd, counts_arrays, base_label="Counts", group_by_3=count_scale_group_by_3)
        
        # X ticks
        axr.set_xticks(x)
        axr.set_xticklabels(tick_labels, rotation=45, ha="right")
    
        # ===== Legend =====

        # Choose legend position dynamically (based on reference curve)
        x_for_score = x
        y_for_score = np.max(np.vstack(top_y_arrays), axis=0)
        leg = cls._place_legend_emptiest_corner(axd, x_for_score, y_for_score, labels=None, fontsize=cls.legend_fontsize)
        
        # ===== Optional Model Spec Annotations =====

        fig.canvas.draw() # legend must be rendered before we use location properties
        leg_bbox_pixels = leg.get_window_extent()
        cls._place_annotation_emptiest_corner(
            fig, axd, x_for_score, y_for_score, labels=None, leg_bbox_pixels=leg_bbox_pixels,
            spec_text=spec_text, spec_table_rows=spec_table_rows, fontsize=cls.legend_fontsize
        )
                        
        # ===== Final/Save =====
        
        sns.despine(ax=axd, trim=True)
        sns.despine(ax=axr, trim=True)
    
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, (axd, axr)
    
    @classmethod
    def plot_dist_and_ratio_discrete_grouped(
        cls,
        *,
        column_name,
        ref_vals,
        comp_vals_dict,             # dict: model_name -> np.array
        model_legend_titles,        # ["Geant4", "model1 label", "model2 label", ...] aligned with model_names
        density=False,
        use_log=False,
        out_file=None,
        title=None,
        ratio_ylim=(0.5, 1.5),
        ratio_bands=(0.05, 0.10, 0.20),
        dynamic_count_scale=True,
        count_scale_group_by_3=False,
        spec_text=None,
        spec_table_rows=None,
        top_k=None,
        sort_descending=False
    ):
        """
        Plots discrete data as a bargraph where all bars are grouped together.
        """

        # ===== Defaults if not set =====
        
        model_names = list(comp_vals_dict.keys())
        palette = sns.color_palette("colorblind", n_colors=max(3, len(model_names) + 1))
        
        feature_name, unit = cls._get_labels(column_name)
        
        # ===== Get data frequency distribution =====
        
        all_unique_particles, real_freq_dist, sampled_freq_dist_dict = plotting_v2._get_data_frequency_distribution(
            in_real_data=ref_vals,
            in_generated_data=comp_vals_dict,
            top_k=top_k,
            sort_descending=sort_descending
        )
        
        # Keep raw counts for error propagation
        real_counts_raw = real_freq_dist.copy()
        sampled_counts_raw_dict = {
            mn: sampled_freq_dist_dict[mn].copy()
            for mn in model_names
        }
        
        # ===== Normalize / top-panel errors =====
        
        if density:
            real_total = real_counts_raw.sum()
            if real_total > 0:
                r_y = real_counts_raw.astype(float) / real_total
                r_err = np.sqrt(real_counts_raw.astype(float)) / real_total
            else:
                r_y = real_counts_raw.astype(float)
                r_err = np.zeros_like(r_y, dtype=float)

            g_y_dict = {}
            g_err_dict = {}
            for mn in model_names:
                g_total = sampled_counts_raw_dict[mn].sum()
                if g_total > 0:
                    g_y_dict[mn] = sampled_counts_raw_dict[mn].astype(float) / g_total
                    g_err_dict[mn] = np.sqrt(sampled_counts_raw_dict[mn].astype(float)) / g_total
                else:
                    g_y_dict[mn] = sampled_counts_raw_dict[mn].astype(float)
                    g_err_dict[mn] = np.zeros_like(g_y_dict[mn], dtype=float)
        else:
            r_y = real_counts_raw.astype(float)
            r_err = np.sqrt(real_counts_raw.astype(float))

            g_y_dict = {
                mn: sampled_counts_raw_dict[mn].astype(float)
                for mn in model_names
            }
            g_err_dict = {
                mn: np.sqrt(sampled_counts_raw_dict[mn].astype(float))
                for mn in model_names
            }
            
        # ===== Ratio denominator always uses shape-only densities =====

        rN = real_counts_raw.sum()
        r_density = real_counts_raw.astype(float) / rN if rN > 0 else np.zeros_like(real_counts_raw, dtype=float)
        # Ratio error propagation: sqrt(N) / N = 1 / sqrt(N)
        r_density_err = np.sqrt(real_counts_raw.astype(float)) / rN if rN > 0 else np.zeros_like(real_counts_raw, dtype=float)
        
        fig, (axd, axr) = plt.subplots(
            2, 1,
            figsize=(6, 6),
            sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
            constrained_layout=True,
        )
        
        # ===== Writing =====

        # Useful for slides etc, where the figure might be standalone
        if title:
            fig.suptitle(title, fontsize=12)

        # Labels
        den_or_cnts = "Density" if density else "Counts"
        axd.set_ylabel(f"log({den_or_cnts})" if use_log else den_or_cnts)
        
        axr.set_ylabel("Gen / Geant4")
        axr.set_xlabel(f"{feature_name} {unit}".rstrip())

        # ===== Axes =====

        axd.set_yscale("log" if use_log else "linear")

        for axd in ([axd]):
            axd.minorticks_on()
            axd.tick_params(which="both", direction="in", top=True, right=True)
        
        all_unique_particles_ticks = [str(int(p)) for p in all_unique_particles]
        plt.xticks(rotation=45, ha='right')
        
        # ===== X positions for grouped bars =====

        n_categories = len(all_unique_particles)
        n_series = 1 + len(model_names)   # Geant4 + generated models
        x = np.arange(n_categories)

        total_group_width = 0.82
        bar_width = total_group_width / n_series
        offsets = (np.arange(n_series) - (n_series - 1) / 2.0) * bar_width

        tick_labels = [str(int(p)) for p in all_unique_particles]

        # For legend scoring / annotations
        top_y_arrays = [r_y]
    
        # ===== Top panel: grouped bars =====
        
        error_kw = dict(linewidth=1, capsize=0, capthick=0)
        
        # For log scale, error bars cannot go to/below zero
        if use_log:
            eps = 1e-12
            if density:
                real_freq_dist = np.clip(real_freq_dist, eps, None)
                for mn in model_names:
                    sampled_freq_dist_dict[mn] = np.clip(sampled_freq_dist_dict[mn], eps, None)
        
        # Do plot
        axd.bar(
            x + offsets[0], r_y, yerr=r_err,
            label=model_legend_titles[0], color='black', linewidth=0, alpha=0.8, width=bar_width, align='center', error_kw=error_kw
        )
        for i, mn in enumerate(model_names, start=1):
            g_y = g_y_dict[mn]
            g_err = g_err_dict[mn]
            top_y_arrays.append(g_y)

            axd.bar(
                x + offsets[i], g_y_dict[mn], yerr=g_err,
                label=model_legend_titles[i], color=palette[i], linewidth=0, alpha=0.8, width=bar_width, align='center', error_kw=error_kw
            )
            
        axd.set_xticks(x)
        axd.set_xticklabels(all_unique_particles_ticks, rotation=45, ha="right")
        
        # ===== Bottom panel: ratios (shape-only densities) =====

        for i, mn in enumerate(model_names, start=1):
            color = palette[i]

            gcounts = sampled_counts_raw_dict[mn].astype(float)
            gN = gcounts.sum()

            g_density = gcounts / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)
            g_density_err = np.sqrt(gcounts) / gN if gN > 0 else np.zeros_like(gcounts, dtype=float)

            ratio = np.divide(
                g_density,
                r_density,
                out=np.full_like(g_density, np.nan, dtype=float),
                where=r_density > 0
            )

            ratio_err = np.full_like(ratio, np.nan, dtype=float)
            ok = (g_density > 0) & (r_density > 0)
            ratio_err[ok] = ratio[ok] * np.sqrt(
                (g_density_err[ok] / g_density[ok])**2 +
                (r_density_err[ok] / r_density[ok])**2
            )

            ok2 = np.isfinite(ratio) & np.isfinite(ratio_err)

            axr.errorbar(
                (x + offsets[i])[ok2],
                ratio[ok2],
                yerr=ratio_err[ok2],
                fmt="o",
                ms=3.5,
                color=color,
                ecolor=color,
                elinewidth=0.8,
                capsize=0,
                alpha=0.95,
                zorder=3
            )
        
        # ===== Aesthetics =====
        
        axd.text(0.015, 0.95, "(a)", transform=axd.transAxes, fontsize=11, va="top")
        axr.text(0.015, 0.95, "(b)", transform=axr.transAxes, fontsize=11, va="top")

        # Ratio guide lines
        axr.axhline(1.0, color="0.25", lw=1.2)
        for frac in ratio_bands:
            axr.axhline(1.0 + frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
            axr.axhline(1.0 - frac, color="0.6", linestyle="--", lw=0.8, alpha=0.6)
        axr.set_ylim(*ratio_ylim)

        # Top-panel headroom
        ymin, ymax = axd.get_ylim()
        if use_log:
            if ymin > 0 and ymax > 0:
                log_range = np.log10(ymax) - np.log10(ymin)
                new_log_top = np.log10(ymax) + 0.10 * log_range
                axd.set_ylim(ymin, 10**new_log_top)
        else:
            axd.set_ylim(ymin, ymax * 1.10)

        # Fix log tick formatting only when needed
        if use_log:
            axd.yaxis.set_major_locator(mticker.LogLocator(base=10))
            axd.yaxis.set_major_formatter(mticker.LogFormatterMathtext(base=10))
        
        # Dynamic y scaling for counts (not density and not log because those do not need it)
        if dynamic_count_scale and (not density) and (not use_log):
            counts_arrays = [real_freq_dist] + [sampled_freq_dist_dict[mn] for mn in model_names]
            cls._apply_dynamic_count_scaling_ticks(axd, counts_arrays, base_label="Counts", group_by_3=count_scale_group_by_3)
        
        # X ticks
        axr.set_xticks(x)
        axr.set_xticklabels(tick_labels, rotation=45, ha="right")
    
        # ===== Legend =====

        # Choose legend position dynamically (based on reference curve)
        x_for_score = x
        y_for_score = np.max(np.vstack(top_y_arrays), axis=0)
        leg = cls._place_legend_emptiest_corner(axd, x_for_score, y_for_score, labels=None, fontsize=cls.legend_fontsize)
        
        # ===== Optional Model Spec Annotations =====

        fig.canvas.draw() # legend must be rendered before we use location properties
        leg_bbox_pixels = leg.get_window_extent()
        cls._place_annotation_emptiest_corner(
            fig, axd, x_for_score, y_for_score, labels=None, leg_bbox_pixels=leg_bbox_pixels,
            spec_text=spec_text, spec_table_rows=spec_table_rows, fontsize=cls.legend_fontsize
        )
                        
        # ===== Final/Save =====
        
        sns.despine(ax=axd, trim=True)
        sns.despine(ax=axr, trim=True)
    
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, (axd, axr)


class plotting_old3:
    """
    All plotting functions will follow a similar API. This allows easy intuitive generation
    of various types of plots.
    
    normalized: (optional: False) bool, should the values be normalized to an area of 1 before plotting?
    use_log: (optional: False) bool, should the dependent axis be log scaled?
    juxtaposed: (optional: False) bool, if input contains multiple values (array) should all be plotted on the same
        axis or should different axes be used side-by-side.
    out_file: (optional) pathlib.Path, file to save figure to. plt.show will always be called since it
        naturally only works if there is a way to show the figures.
    """
    
    # Preferred starting colors
    first_colors = ['blue', 'orange', 'purple', 'red', 'green']
    # Build color list (done right inside the class body)
    _all_colors = list(mcolors.CSS4_COLORS.keys())
    _rgb_colors = [(c, mcolors.to_rgb(c)) for c in _all_colors]
    # Filter out very bright / whiteish colors (value < 0.9)    x
    _filtered_colors = [
        name for name, rgb in _rgb_colors
        if mcolors.rgb_to_hsv(rgb)[2] < 0.96
    ]
    # Remove duplicates of first colors
    _filtered_colors = [c for c in _filtered_colors if c not in ['blue', 'orange', 'purple', 'red', 'green', 'black']]
    # Final palette
    colors = ['blue', 'orange', 'purple', 'red', 'green'] + _filtered_colors
    # Apply globally to Matplotlib
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colors)
    
    default_figsize = (21, 6)
    default_dpi = 300
    distributions_per_row = 3
    legend_items_per_col = 10
    
    verbose_columns = ["pdgid", "e", "px", "py", "pz", "pt", "eta", "theta", "phi"]
    
    def __init__(self):
        # We store the data for the distributions here so it can be reused
        # Format: dict(mode_name, data)
        self.all_real_data = {}
        self.all_sampled_data = {}

    def load_data_by_model_names(self, model_names):
        for model_name in model_names:
            real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
            sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
            self.all_real_data[model_name] = real_verbose_data
            self.all_sampled_data[model_name] = sampled_verbose_data
    
    def ensure_data_loaded(self, model_names, real_data=False):
        "Loads data for model_names if not already loaded. Nothing otherwise."
        for model_name in model_names:
            if real_data == True and model_name not in self.all_real_data:
                real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
                self.all_real_data[model_name] = real_verbose_data
            if model_name not in self.all_sampled_data:
                sampled_verbose_data = data_manager.load_verbose_dataset(pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples_verbose.csv', pad_token = np.nan)
                self.all_sampled_data[model_name] = sampled_verbose_data

    """
    Plotting training runs and distributions of leading particles.
    """
    
    # learning rate decay scheduler
    @staticmethod
    def _get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters, min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult):
        if lr_scheduler == 'cosine_annealing_with_warmup':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 2) if it > lr_decay_iters, return min learning rate
            if it >= lr_decay_iters:
                return min_lr
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        elif lr_scheduler == 'cosine_with_warmup':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * it / warmup_iters
            # 3) in between, use cosine decay down to min learning rate
            decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        elif lr_scheduler == 'cosine_annealing_with_warm_restarts':
            # 1) linear warmup for warmup_iters steps
            if it < warmup_iters:
                return learning_rate * (it / warmup_iters)
            # Adjust iteration to account for warmup
            it -= warmup_iters
            # 2) Find current cycle and  in the cycle
            cycle = 0
            curr_cycle_len = lr_decay_iters
            iter_in_cycle = it
            while iter_in_cycle >= curr_cycle_len:
                iter_in_cycle -= curr_cycle_len
                cycle += 1
                curr_cycle_len = int(curr_cycle_len * cycle_steps_mult)
            # 3) Decay the base learning rate for the current cycle
            curr_base_lr = learning_rate * (base_lr_decay_mult ** cycle)
            # 4) Normalized progress within the cycle
            t = iter_in_cycle / curr_cycle_len
            # 5) Cosine annealing
            lr = min_lr + 0.5 * (curr_base_lr - min_lr) * (1 + math.cos(math.pi * t))
            return lr
        raise ValueError(f"Unknown lr_scheduler {lr_scheduler}")
    
    @staticmethod
    def plot_training_run(model_names, y_lim=None, x_lim=None, use_log=False, out_file=None, plot_lr_schedule=False):
        """
        Wrapper to plot a training run. Handles plotting lines and markers for training and validation loss.
        """
        
        if not isinstance(model_names, list):
            model_names = [model_names]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if len(model_names) < 3:
            fig.suptitle(f'Training Progress for {model_names}', fontsize=16)
        else:
            fig.suptitle(f'Training Progress for various models', fontsize=16)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
            
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if use_log:
            ax.set_yscale('log')
            
        for idx, model_name in enumerate(model_names):
            # Parse model data
            model_data = tables.get_all_data(model_name)
            min_val_row = model_data.checkpointed_df.loc[model_data.checkpointed_df['val_loss'].idxmin()]
            final_row = model_data.running_df.iloc[-1]
            
            if plot_lr_schedule:
                config_filepath = pUtil.get_model_config_filepath(model_name)
                with open(config_filepath, 'r') as f:
                    model_config = json.load(f)
                    training_conf = model_config.get('training_config', {})
                    lr_scheduler = training_conf.get('lr_scheduler', 'cosine_annealing_with_warmup')
                    warmup_iters = training_conf .get('warmup_iters', 0)
                    lr_decay_iters = training_conf .get('lr_decay_iters', 0)
                    learning_rate = training_conf .get('learning_rate', 0)
                    min_lr = training_conf .get('min_lr', 0)
                    base_lr_decay_mult = training_conf .get('base_lr_decay_mult', 1)
                    cycle_steps_mult = training_conf .get('cycle_steps_mult', 1)
                iters = [i for i in range(model_data.running_df['iter'].max())]
                lrs = [plotting._get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters, min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult) for it in iters]
                
                # Another y-axis for lr as otherwise it would be too small to see 
                lrax = ax.twinx()
                lrax.plot(iters, lrs, label=f'Learning rate', color="magenta", linestyle='solid', linewidth=2)
                lrax.tick_params(axis='y', labelcolor="magenta")
                # Decimal can be confusing so we switch to scientific
                lrax.yaxis.set_major_formatter(FuncFormatter(sci_notation))

            # Do plot
            ax.plot(model_data.running_df['iter'], model_data.running_df['train_loss'], label=f'Training Loss ({model_name})', color=plotting.colors[idx], linestyle='solid', linewidth=2)
            ax.plot(model_data.running_df['iter'], model_data.running_df['val_loss'], label=f'Validation Loss ({model_name})', color=plotting.colors[idx], linestyle='solid', linewidth=2)
            ax.scatter(min_val_row['iter'], min_val_row['train_loss'], label=f'Min Saved Train Loss ({model_name}; {min_val_row["train_loss"]:.4f})', color=plotting.colors[idx], marker='s')
            ax.scatter(min_val_row['iter'], min_val_row['val_loss'], label=f'Min Saved Val Loss ({model_name}; {min_val_row["val_loss"]:.4f})', color=plotting.colors[idx], marker='o')
            # ax.annotate(model_name, xy=(final_row['iter'], final_row['val_loss']), xytext=(final_row['iter'] * 1.005, final_row['val_loss'] - 0.02), fontsize=9, color=plotting.colors[idx])

        # Final touches and show and/or save
        if len(model_names) > 3:
            ax.legend(loc='upper right', fontsize=12)
        else:
            ax.legend(loc='best', fontsize=12)
        fig.tight_layout()
        ax.grid()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    @staticmethod
    def plot_validation_run(model_names, model_legend_titles=None, y_lim=None, x_lim=None, use_log=False, out_file=None, plot_lr_schedule=False):
        """
        Wrapper to plot a training run. Handles plotting lines and markers for training and validation loss.
        """
        
        if not isinstance(model_names, list):
            model_names = [model_names]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        if len(model_names) < 3:
            fig.suptitle(f'Training Progress for {model_names}', fontsize=16)
        else:
            fig.suptitle(f'Training Progress for various models', fontsize=16)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        ax.set_xlabel("Iteration", fontsize=16)
        ax.set_ylabel("Validation Loss", fontsize=16)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if use_log:
            ax.set_yscale('log')
            
        for idx, model_name in enumerate(model_names):
            # Parse model data
            model_data = tables.get_all_data(model_name)
            min_val_row = model_data.checkpointed_df.loc[model_data.checkpointed_df['val_loss'].idxmin()]
            final_row = model_data.running_df.iloc[-1]
            
            if plot_lr_schedule:
                config_filepath = pUtil.get_model_config_filepath(model_name)
                with open(config_filepath, 'r') as f:
                    model_config = json.load(f)
                    training_conf = model_config.get('training_config', {})
                    lr_scheduler = training_conf.get('lr_scheduler', 'cosine_annealing_with_warmup')
                    warmup_iters = training_conf .get('warmup_iters', 0)
                    lr_decay_iters = training_conf .get('lr_decay_iters', 0)
                    learning_rate = training_conf .get('learning_rate', 0)
                    min_lr = training_conf .get('min_lr', 0)
                    base_lr_decay_mult = training_conf .get('base_lr_decay_mult', 1)
                    cycle_steps_mult = training_conf .get('cycle_steps_mult', 1)
                iters = [i for i in range(model_data.running_df['iter'].max())]
                lrs = [plotting._get_lr(it, lr_scheduler, warmup_iters, lr_decay_iters, min_lr, learning_rate, cycle_steps_mult, base_lr_decay_mult) for it in iters]
                
                # Another y-axis for lr as otherwise it would be too small to see 
                lrax = ax.twinx()
                lrax.plot(iters, lrs, label=f'Learning rate', color="magenta", linestyle='solid', linewidth=2)
                lrax.tick_params(axis='y', labelcolor="magenta")
                # Decimal can be confusing so we switch to scientific
                lrax.yaxis.set_major_formatter(FuncFormatter(sci_notation))

            # If a name is specified use that.
            label_name = model_name
            if model_legend_titles is not None and idx < len(model_legend_titles):
                label_name = model_legend_titles[idx]
                
            # Do plot
            ax.plot(model_data.running_df['iter'], model_data.running_df['val_loss'], label=f'{label_name}', color=plotting.colors[idx], linestyle='solid', linewidth=2)
            # ax.annotate(label_name, xy=(final_row['iter'], final_row['val_loss']), xytext=(final_row['iter'] * 1.005, final_row['val_loss'] - 0.02), fontsize=14, color=plotting.colors[idx])

        # Final touches and show and/or save
        ax.legend(loc='upper right', fontsize=16, ncol=int(len(model_names) // (plotting.legend_items_per_col + 1)) + 1)
        fig.tight_layout()
        ax.grid()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    @staticmethod
    def get_common_data(model_name ):
        """
        Retrieves bin widths and ranges for each feature and the real and sampled leading particles dataframes.
        """
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
        dictionary = Dictionary(dictionary_filename)
        
        def get_bin_count(type_str):
            step_size = dictionary.token_step_size(type_str)
            if type_str in ['eta', 'theta', 'phi']:
                step_size = 0.1
            if type_str == 'theta':
                theta_min = 0 if dictionary.token_min('theta') == 0 else dictionary.token_min('theta')
                theta_max = np.pi if dictionary.token_max('theta') == 0 else dictionary.token_max('theta')
                return int((theta_max - theta_min) // step_size)
            if type_str == 'phi':
                phi_min = -np.pi if dictionary.token_min('phi') == 0 else dictionary.token_min('phi')
                phi_max = np.pi if dictionary.token_max('phi') == 0 else dictionary.token_max('phi')
                return int((phi_max - phi_min) // step_size)
            return int(dictionary.token_range(type_str) // step_size)
        
        # For now, I have replaced those with hard coded values.
        theta_min = 0 if dictionary.token_min('theta') == 0 else dictionary.token_min('theta')
        theta_max = np.pi if dictionary.token_max('theta') == 0 else dictionary.token_max('theta')
        phi_min = -np.pi if dictionary.token_min('phi') == 0 else dictionary.token_min('phi')
        phi_max = np.pi if dictionary.token_max('phi') == 0 else dictionary.token_max('phi')
        
        bin_settings = {
            "num_particles": { "min": -0.5,                          "max": 50.5,                          "bins": 51 },
            "e":             { "min": 0,                             "max": 35000,                         "bins": 70 },
            "px":            { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "py":            { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "pz":            { "min": -5000,                         "max": 35000,                         "bins": 70 },
            "eta":           { "min": dictionary.token_min('eta'),   "max": dictionary.token_max('eta'),   "bins": get_bin_count('eta') },
            "theta":         { "min": theta_min,                     "max": theta_max,                     "bins": get_bin_count('theta') },
            "phi":           { "min": phi_min,                       "max": phi_max,                       "bins": get_bin_count('phi') },
            "pt":            { "min": dictionary.token_min('pt'),    "max": dictionary.token_max('pt'),    "bins": 70 },
        }

        return bin_settings

    def _do_plot_get_labels(self, column_name):
        # assert column_name in plotting.verbose_columns, f"Invalid column name: {column_name}. Must be one of {plotting.verbose_columns}."
        
        if column_name == 'num_particles':
            feature_name = 'Number of particles'
            unit = ''
            return feature_name, unit
        
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '[MeV]'
                        
        feature_name = column_name
        if column_name == 'pt':
            feature_name = r'$p_{T}$'
        elif column_name == 'px':
            feature_name = r'$p_{x}$'
        elif column_name == 'py':
            feature_name = r'$p_{y}$'
        elif column_name == 'pz':
            feature_name = r'$p_{z}$'
        elif column_name == 'eta':
            feature_name = r'$\eta$'
        elif column_name == 'theta':
            feature_name = r'$\theta$'
        elif column_name == 'phi':
            feature_name = r'$\phi$'
        return feature_name, unit

    def _do_plot(self, data, title=None, normalized=False, use_log=False, out_file=None):
        """
        data: must be a dict with elements of form (model_name, column_name, data_real, data_generated)
              The first element is expected to be the Geant4 Simulation.
        """

        # 2 rows because we have a ratio on the bottom
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True, dpi=plotting.default_dpi)
        fig.suptitle(title, fontsize=16)
        
        axd, axr = axes
        # Since the column name will be the same for all data, we can just extract it here.
        # Further, the input data will also be the same for all of them.
        _, f_column_name, f_real_data, _ = data[0]
        feature_name, unit = self._do_plot_get_labels(f_column_name)
        axd.set_ylabel('log(Frequency)' if use_log else 'Frequency', fontsize=16)
        axd.set_yscale('log' if use_log else 'linear')
        axr.set_ylabel('Generated / Geant4', fontsize=16)
        axr.set_xlabel(f'{feature_name} {unit}', fontsize=16)
        
        # Will be same across all models
        bin_settings = plotting.get_common_data(model_name)
        range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
        n_bins = bin_settings[column_name]['bins']

        rcounts, redges = np.histogram(data_real, bins=n_bins, range=range, density=normalized)
        input_total_counts = np.sum(rcounts)
        
        axd.step(redges[:-1], rcounts, where='post', label=f'Geant4 Simulation  = {input_total_counts}', color='black', linewidth=2)
        axd.errorbar(bin_centers, rcounts, yerr=rerrors, fmt='none', ecolor='black', elinewidth=0.5, capsize=0)
        
        for model_name, column_name, data_real, data_generated in data:
            gcounts, gedges = np.histogram(data_generated, bins=n_bins, range=range, density=normalized)
            generated_total_counts = np.sum(gcounts)

            # Top pannel, the distribution
            
            # Bin centers for error bars
            bin_centers = (redges[:-1] + redges[1:]) / 2
            bin_width = redges[1] - redges[0]
            
            # Statistical uncertainties
            rerrors = np.sqrt(rcounts)
            gerrors = np.sqrt(gcounts)

            if normalized:
                rerrors = rerrors / input_total_counts
                gerrors = gerrors / generated_total_counts
            
            axd.step(gedges[:-1], gcounts, where='post', label=f'Generated ({model_name})  = {generated_total_counts}', color=plotting.colors[1], linewidth=2)
            axd.errorbar(bin_centers, gcounts, yerr=gerrors, fmt='none', ecolor=plotting.colors[1], elinewidth=0.5, capsize=0)
        
            # Bottom pannel, the ratio
            
            input_density = rcounts / input_total_counts
            generated_density = gcounts / generated_total_counts
            ratio = np.divide(generated_density, input_density, out=np.zeros_like(generated_density), where=input_density!=0)
            
            axr.stairs(ratio, redges, label=f'Ratio ({model_name})', color=plotting.colors[1], linewidth=1)
            # Reasonable range for ratios around 1
            axr.set_ylim(0.5, 1.5)
            for frac in [0.05, 0.10, 0.20]:
                axr.axhline(y=1.0 + frac, color="gray", linestyle="--", lw=1)
                axr.axhline(y=1.0 - frac, color="gray", linestyle="--", lw=1)
            
        axd.legend(loc='best', fontsize=12)

        # Finishing touches and show and/or save
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        return fig, axes
    
    def _do_plot_single(self, data, title=None, normalized=False, use_log=False, out_file=None):
        """
        data: must be a dict with elements of form (model_name, column_name, data_real, data_generated)
              The first element is expected to be the Geant4 Simulation.
        """

        # 2 rows because we have a ratio on the bottom
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]}, constrained_layout=True, dpi=plotting.default_dpi)
        fig.suptitle(title, fontsize=16)
        
        axd, axr = axes
        
        model_name, column_name, data_real, data_generated = data[0]
        bin_settings = plotting.get_common_data(model_name)
        range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
        n_bins = bin_settings[column_name]['bins']
        
        rcounts, redges = np.histogram(data_real, bins=n_bins, range=range, density=normalized)
        gcounts, gedges = np.histogram(data_generated, bins=n_bins, range=range, density=normalized)
        
        input_total_counts = np.sum(rcounts)
        generated_total_counts = np.sum(gcounts)

        # Top pannel, the distribution
        feature_name, unit = self._do_plot_get_labels(column_name)
        axd.set_ylabel('log(Frequency)' if use_log else 'Frequency', fontsize=16)
        axd.set_yscale('log' if use_log else 'linear')
        
        # Bin centers for error bars
        bin_centers = (redges[:-1] + redges[1:]) / 2
        bin_width = redges[1] - redges[0]
        
        # Statistical uncertainties
        rerrors = np.sqrt(rcounts)
        gerrors = np.sqrt(gcounts)

        if normalized:
            rerrors = rerrors / input_total_counts
            gerrors = gerrors / generated_total_counts
        
        axd.step(redges[:-1], rcounts, where='post', label=f'Geant4 Simulation  = {input_total_counts}', color='black', linewidth=2)
        axd.step(gedges[:-1], gcounts, where='post', label=f'Generated ({model_name})  = {generated_total_counts}', color=plotting.colors[1], linewidth=2)
        axd.errorbar(bin_centers, rcounts, yerr=rerrors, fmt='none', ecolor='black', elinewidth=0.5, capsize=0)
        axd.errorbar(bin_centers, gcounts, yerr=gerrors, fmt='none', ecolor=plotting.colors[1], elinewidth=0.5, capsize=0)
        axd.legend(loc='best', fontsize=12)
        
        # Bottom pannel, the ratio
        input_density = rcounts / input_total_counts
        generated_density = gcounts / generated_total_counts
        ratio = np.divide(generated_density, input_density, out=np.zeros_like(generated_density), where=input_density!=0)
        
        axr.stairs(ratio, redges, label=f'Ratio ({model_name})', color=plotting.colors[1], linewidth=1)
        # Reasonable range for ratios around 1
        axr.set_ylim(0.5, 1.5)
        for frac in [0.05, 0.10, 0.20]:
            axr.axhline(y=1.0 + frac, color="gray", linestyle="--", lw=1)
            axr.axhline(y=1.0 - frac, color="gray", linestyle="--", lw=1)
        
        axr.set_xlabel(f'{feature_name} {unit}', fontsize=16)
        axr.set_ylabel('Generated / Geant4', fontsize=16)

        # Finishing touches and show and/or save
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        return fig, axes

    def plot_distribution_leading(self, model_names, column_name=None, normalized=False, use_log=False, out_file=None):
        data = []
        for model_name in model_names:
            relevant_column_pos = plotting.verbose_columns.index(column_name)
            real_verbose_data = self.all_real_data[model_name]
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.nanargmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
            
            sampled_verbose_data = self.all_sampled_data[model_name]
            all_instances_of_this_column_sampled = []
            for event in sampled_verbose_data:
                secondaries = event[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.nanargmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
            
            data.append((model_name, column_name, all_instances_of_this_column_real, all_instances_of_this_column_sampled))

        feature_name, _ = self._do_plot_get_labels(column_name)
        self._do_plot(data, title=f'Distribution of {feature_name} for Leading Particles', normalized=normalized, use_log=use_log, out_file=out_file)

    def plot_distribution_all(self, model_names, column_name=None, normalized=False, use_log=False, out_file=None):
        data = []
        for model_name in model_names:
            relevant_column_pos = plotting.verbose_columns.index(column_name)
            real_verbose_data = self.all_real_data[model_name]
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_real.append(particle[relevant_column_pos])
            
            sampled_verbose_data = self.all_sampled_data[model_name]
            all_instances_of_this_column_sampled = []
            for event in sampled_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_sampled.append(particle[relevant_column_pos])
                        
            data.append((model_name, column_name, all_instances_of_this_column_real, all_instances_of_this_column_sampled))

        feature_name, _ = self._do_plot_get_labels(column_name)
        self._do_plot(data, title=f'Distribution of {feature_name} for All Particles', normalized=normalized, use_log=use_log, out_file=out_file)
            
    def plot_pdgid_distribution_leading(self, model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        relevant_column_pos = plotting.verbose_columns.index('pdgid')
        real_verbose_data = self.all_real_data[model_name]
        all_instances_of_this_column_real = []
        for event in real_verbose_data:
            secondaries = event[1:]
            # Find index of particle with the highest energy
            leading_particle_idx = np.nanargmax(secondaries[:, 1])
            leading_particle = secondaries[leading_particle_idx]
            all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
        
        sampled_verbose_data = self.all_sampled_data[model_name]
        all_instances_of_this_column_sampled = []
        for event in sampled_verbose_data:
            secondaries = event[1:]
            # Find index of particle with the highest energy
            leading_particle_idx = np.nanargmax(secondaries[:, 1])
            leading_particle = secondaries[leading_particle_idx]
            all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
            
        # real_pdgids, sampled_pdgids = real_df['pdgid'], sampled_df['pdgid']
        real_freq, sampled_freq = Counter(all_instances_of_this_column_real), Counter(all_instances_of_this_column_sampled)
        
        # Union of all particle labels from both histograms
        all_particles = sorted(set(real_freq.keys()).union(sampled_freq.keys()))
        # Sorting them by frequency in real leading particles to ensure a legible plot
        sorted_particles = sorted(all_particles, key=lambda p: real_freq[p], reverse=True)
        # Build aligned values for both histograms
        real_values = [real_freq.get(p, 0) for p in sorted_particles]
        sampled_values = [sampled_freq.get(p, 0) for p in sorted_particles]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        fig.suptitle(f'Particle Type Distributions {"(Normalized)" if normalized else ""}')
        fig.supxlabel('Particle Type')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        if use_log:
            ax.set_yscale('log')
        if normalized:
            total_real = sum(real_values)
            total_sampled = sum(sampled_values)
            real_values = [v / total_real for v in real_values]
            sampled_values = [v / total_sampled for v in sampled_values]
        
        # Do plot
        ax.bar(range(len(sorted_particles)), real_values, label=f'Geant4 Simulation', color='black', alpha=0.7, width=0.9, align='center')
        ax.bar(range(len(sorted_particles)), sampled_values, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7, width=0.9, align='center')
        ax.set_xticks(range(len(sorted_particles)), sorted_particles, rotation=45, ha='right')
        
        # Finishing touches and show and/or save
        plt.legend(loc='best', fontsize=12)
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax

    def plot_pdgid_distribution_all(self, model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        meta_data = tables.get_meta_data(model_name)
        
        testing_bin_filename = pUtil.get_model_preparation_dir(model_name) / 'test_real.bin'
        generated_samples_filename = pUtil.get_latest_sampling_dir(model_name) / 'untokenized_samples.csv'
        
        num_tokens_per_particle_raw = 5
        testing_real_data = np.memmap(testing_bin_filename, dtype=np.float64, mode='r')
        testing_real_data = testing_real_data.reshape(-1, int((meta_data.max_sequence_length - 2) / num_tokens_per_particle_raw), num_tokens_per_particle_raw)
        generated_sample_data = data_manager.load_geant4_dataset(generated_samples_filename, pad_token=0.0)
        
        real_pdgid_freq_dist, _ = dataset.get_pdgid_frequency_distribution(testing_real_data)
        sampled_pdgid_freq_dist, _ = dataset.get_pdgid_frequency_distribution(generated_sample_data)
        real_freq, sampled_freq = real_pdgid_freq_dist, sampled_pdgid_freq_dist
        
        # Union of all particle labels from both histograms
        all_particles = sorted(set(real_freq.keys()).union(sampled_freq.keys()))
        # Sorting them by frequency in real leading particles to ensure a legible plot
        sorted_particles = sorted(all_particles, key=lambda p: real_freq[p], reverse=True)
        # Build aligned values for both histograms
        real_values = [real_freq.get(p, 0) for p in sorted_particles]
        sampled_values = [sampled_freq.get(p, 0) for p in sorted_particles]
        
        # Set up plot
        fig, ax = plt.subplots(figsize=plotting.default_figsize, dpi=plotting.default_dpi)
        fig.suptitle(f'Particle Type Distributions {"(Normalized)" if normalized else ""}')
        fig.supxlabel('Particle Type')
        fig.supylabel('log(Frequency)' if use_log else 'Frequency')
        if use_log:
            ax.set_yscale('log')
        if normalized:
            total_real = sum(real_values)
            total_sampled = sum(sampled_values)
            real_values = [v / total_real for v in real_values]
            sampled_values = [v / total_sampled for v in sampled_values]
        
        # Do plot
        ax.bar(range(len(sorted_particles)), real_values, label=f'Geant4 Simulation', color='black', alpha=0.7, width=0.9, align='center')
        ax.bar(range(len(sorted_particles)), sampled_values, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7, width=0.9, align='center')
        ax.set_xticks(range(len(sorted_particles)), sorted_particles, rotation=45, ha='right')
        
        # Finishing touches and show and/or save
        plt.legend(loc='best', fontsize=12)
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
        return fig, ax
    
    def plot_energy_conservation(self, model_names, normalized=False, use_log=False, out_file=None):
        model_name = model_names[0]
        
        MASS_CARBON = 931.5 * 12 # MeV
        generated_samples_data = self.all_sampled_data[model_name]
        generated_samples_data = np.nan_to_num(generated_samples_data, copy=True, nan=0.0)

        # Perform computation
        dictionary_filename = pUtil.get_model_preparation_dir(model_name) / 'dictionary.json'
        dictionary = Dictionary(dictionary_filename)
        
        pdgid_mass_dict = {}
        for idx, pdgid in dictionary.pdgids.items():
            if pdgid == 0:
                continue
            part = Particle.from_pdgid(pdgid)
            pdgid_mass_dict[pdgid] = part.mass

        computed_data = np.full(shape=(len(generated_samples_data), 2), fill_value=np.nan)
        for idx, event in enumerate(generated_samples_data):
            # Input vector
            in_particle_vec = vector.obj(mass=pdgid_mass_dict[event[0][0]], px=event[0][2], py=event[0][3], pz=event[0][4])
            in_material_vec = vector.obj(mass=MASS_CARBON, px=0.0, py=0.0, pz=0.0)
            in_vec = in_particle_vec + in_material_vec
            
            # Output vector
            out_vec = vector.obj(mass=0.0, px=0.0, py=0.0, pz=0.0)
            for particle in event[1:]:
                if particle[0] == 0.0:
                    continue
                i_vec = vector.obj(mass=pdgid_mass_dict[particle[0]], px=particle[2], py=particle[3], pz=particle[4])
                out_vec += i_vec
            
            computed_data[idx] = in_vec.e, out_vec.e
        
        # Set up plot
        num_horizontal, num_vertical = min(len(model_names), plotting.distributions_per_row), (math.ceil(1 / plotting.distributions_per_row))
        fig, axes = plt.subplots(num_vertical, num_horizontal, figsize=(8 * num_horizontal, 6 * num_vertical), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'Energy conservation for {model_names}')
        fig.supxlabel(f'Delta Energy (MeV)')
        fig.supylabel('Frequency')
        
        for ax, model_name in zip([axes], model_names):
            # Do plot
            ax.set_yscale('log' if use_log else 'linear')
            ax.hist(computed_data[:,0], bins=50, density=normalized, label=f'Geant4 Simulation', color='black', alpha=0.7)
            ax.hist(computed_data[:,1], bins=50, density=normalized, label=f'Sampled ({model_name})', color=plotting.colors[1], alpha=0.7)

        # Finishing touches and show and/or save
        plt.legend(loc='best', fontsize=12)
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        fig.show()
        
    def plot_num_particles(self, model_names, normalized=False, use_log=False, out_file=None):
        data = []
        for model_name in model_names:
            real_verbose_data = data_manager.load_verbose_dataset(pUtil.get_model_preparation_dir(model_name) / 'real_verbose_test_particles.csv', pad_token = np.nan)
            real_num_particle_data = np.full(shape=(len(real_verbose_data)), fill_value=np.nan)
            for idx, event in enumerate(real_verbose_data):
                secondaries = event[1:]
                secondaries = [secondary for secondary in secondaries if not np.isnan(secondary[0])]
                num_secondaries = len(secondaries)
                real_num_particle_data[idx] = num_secondaries
            
            generated_samples_data = data_manager.load_geant4_dataset(pUtil.get_latest_sampling_dir(model_names[0]) / 'untokenized_samples.csv', pad_token = np.nan)
            sampled_num_particle_data = np.full(shape=(len(generated_samples_data)), fill_value=np.nan)
            for idx, event in enumerate(generated_samples_data):
                secondaries = event[1:]
                secondaries = [secondary for secondary in secondaries if not np.isnan(secondary[0])]
                num_secondaries = len(secondaries)
                sampled_num_particle_data[idx] = num_secondaries
                
            data.append((model_name, 'num_particles', real_num_particle_data, sampled_num_particle_data))
        
        self._do_plot(data, title=f'Distribution of Number of Particles for All Particles', normalized=normalized, use_log=use_log, out_file=out_file)

    # IMPORTANT: This function assumes ALL provided models are from the same dataset!!!!
    def compare_distributions_all_bak(self, model_names, model_legend_titles=None, column_name=None, normalized=False, use_log=False, out_file=None, auto_show=True):
        assert column_name in plotting.verbose_columns, f"Invalid column name: {column_name}. Must be one of {plotting.verbose_columns}."
        unit = ''
        if column_name in ['e', 'pt', 'px', 'py', 'pz']:
            unit = '(MeV)'
        
        unit_label = column_name
        if column_name == 'eta':
            unit_label = 'η'
        elif column_name == 'theta':
            unit_label = 'θ'
        elif column_name == 'phi':
            unit_label = 'φ'
            
        # Set up plot
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), sharex=False, sharey=True, dpi=plotting.default_dpi)
        fig.suptitle(f'{unit_label} Distribution for All Outgoing Particles {"(Normalized)" if normalized else ""}')
        fig.supxlabel(f'{unit_label} {unit}', fontsize=16)
        fig.supylabel('log(Frequency)' if use_log else 'Frequency', fontsize=16)
        
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        
        # This data will be grabbed from the first model to maintain consistency
        bin_settings = plotting.get_common_data(model_names[0])
        range = (bin_settings[column_name]['min'], bin_settings[column_name]['max'])
        n_bins = bin_settings[column_name]['bins']
        
        relevant_column_pos = plotting.verbose_columns.index(column_name)
        
        # The input data
        real_verbose_data = self.all_real_data[model_names[0]]
        all_instances_of_this_column_real = []
        for event in real_verbose_data:
            secondaries = event[1:]
            for particle in secondaries:
                if not np.isnan(particle[relevant_column_pos]):
                    all_instances_of_this_column_real.append(particle[relevant_column_pos])
        
        # The sampled datas 
        sampled_columns_instances = {}
        for model_name in model_names:
            sampled_verbose_data = self.all_sampled_data[model_name]
            all_instances_of_this_column_sampled = []
            for event in sampled_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_sampled.append(particle[relevant_column_pos])
            sampled_columns_instances[model_name] = all_instances_of_this_column_sampled
            
        ax.set_yscale('log' if use_log else 'linear')
        
        base_dataset_label = model_legend_titles[0] if model_legend_titles != None and len(model_legend_titles) > 0 else 'Geant4 Simulation'
        ax.hist(all_instances_of_this_column_real, range=range, bins=n_bins, density=normalized, label=base_dataset_label, color='black', histtype='step', linewidth=2)
        for idx, (model_name, sampled_column_instance) in enumerate(sampled_columns_instances.items()):
            model_label = model_legend_titles[idx + 1] if model_legend_titles != None and idx + 1 < len(model_legend_titles) else model_name
            ax.hist(sampled_column_instance, range=range, bins=n_bins, density=normalized, label=model_label, color=plotting.colors[idx], histtype='step', linewidth=2)
            
        # Finishing touches and show and/or save
        plt.legend(loc='best', fontsize=16)
        fig.tight_layout()
        if out_file != None:
            fig.savefig(out_file, bbox_inches='tight')
        if auto_show == True:
            fig.show()
        return fig, ax
    
    # IMPORTANT: This function assumes ALL provided models are from the same dataset!!!!
    def compare_distributions_all(self, model_names, model_legend_titles=None, column_name=None, normalized=False, use_log=False, out_file=None, auto_show=True, in_ax=None):
        relevant_column_pos = plotting.verbose_columns.index(column_name)
        
        data = []
        for model_name in model_names:
            # The input data
            real_verbose_data = self.all_real_data[model_names[0]]
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                for particle in secondaries:
                    if not np.isnan(particle[relevant_column_pos]):
                        all_instances_of_this_column_real.append(particle[relevant_column_pos])
            
            # The sampled datas 
            sampled_columns_instances = {}
            for model_name in model_names:
                sampled_verbose_data = self.all_sampled_data[model_name]
                all_instances_of_this_column_sampled = []
                for event in sampled_verbose_data:
                    secondaries = event[1:]
                    for particle in secondaries:
                        if not np.isnan(particle[relevant_column_pos]):
                            all_instances_of_this_column_sampled.append(particle[relevant_column_pos])
                sampled_columns_instances[model_name] = all_instances_of_this_column_sampled
                
            data.append((model_name, column_name, all_instances_of_this_column_real, all_instances_of_this_column_sampled))
        
        feature_name, _ = self._do_plot_get_labels(column_name)
        self._do_plot(data, title=f'Distribution of {feature_name} for All Particles', normalized=normalized, use_log=use_log, out_file=out_file)         

    # IMPORTANT: This function assumes ALL provided models are from the same dataset!!!!
    def compare_distributions_leading(self, model_names, model_legend_titles=None, column_name=None, normalized=False, use_log=False, out_file=None, auto_show=True, in_ax=None):
        relevant_column_pos = plotting.verbose_columns.index(column_name)
        
        data = []
        for model_name in model_names:
            # The input data
            real_verbose_data = self.all_real_data[model_names[0]]
            all_instances_of_this_column_real = []
            for event in real_verbose_data:
                secondaries = event[1:]
                # Find index of particle with the highest energy
                leading_particle_idx = np.nanargmax(secondaries[:, 1])
                leading_particle = secondaries[leading_particle_idx]
                all_instances_of_this_column_real.append(leading_particle[relevant_column_pos])
                
            # The sampled datas
            sampled_columns_instances = {}
            for model_name in model_names:
                sampled_verbose_data = self.all_sampled_data[model_name]
                all_instances_of_this_column_sampled = []
                for event in sampled_verbose_data:
                    secondaries = event[1:]
                    # Find index of particle with the highest energy
                    leading_particle_idx = np.nanargmax(secondaries[:, 1])
                    leading_particle = secondaries[leading_particle_idx]
                    all_instances_of_this_column_sampled.append(leading_particle[relevant_column_pos])
                sampled_columns_instances[model_name] = all_instances_of_this_column_sampled
                
            data.append((model_name, column_name, all_instances_of_this_column_real, all_instances_of_this_column_sampled))
        
        feature_name, _ = self._do_plot_get_labels(column_name)
        self._do_plot(data, title=f'Distribution of {feature_name} for All Particles', normalized=normalized, use_log=use_log, out_file=out_file)

class tables:
    """
    This class primarily aggregates all the data from the model meta, config, training and metrics files into various useful formats.
    """
    
    model_metadata_columns          = ['vocab_size', 'max_sequence_length', 'num_train_tokens', 'num_val_tokens']
    model_config_columns            = ['batch_size', 'block_size', 'learning_rate', 'min_lr', 'lr_decay_iters', 'lr_scheduler', 'n_layer', 'n_head', 'n_embd', 'num_params']
    model_training_columns          = ['iters_trained', 'iters_saved', 'min_saved_train_loss', 'min_saved_val_loss']
    model_metrics_columns           = ['coverage', 'mmd', 'kpd_median', 'fpd_value', 'w1m_score', 'w1p_avg_eta', 'w1p_avg_phi', 'w1p_avg_pt']
    model_metrics_columns_verbose   = ['kpd_error', 'fpd_error', 'w1m_score_std', 'w1p_avg_eta_std', 'w1p_avg_phi_std', 'w1p_avg_pt_std']
    model_all_columns               = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns
    model_all_columns_verbose       = ['model_name'] + model_metadata_columns + model_config_columns + model_training_columns + model_metrics_columns + model_metrics_columns_verbose
    
    @staticmethod
    def get_meta_data(model_name):
        meta_filename = pUtil.get_model_meta_filepath(model_name)
        if not meta_filename.exists():
            return None
        
        with open(meta_filename, 'rb') as meta_file:
            meta_data = pickle.load(meta_file)
        
        return SimpleNamespace(
            vocab_size              = meta_data.get('vocab_size', np.nan),
            max_sequence_length     = meta_data.get('max_sequence_length', np.nan),
            num_train_tokens        = meta_data.get('num_train_tokens', np.nan),
            num_val_tokens          = meta_data.get('num_val_tokens', np.nan)
        )
    
    @staticmethod
    def get_config_data(model_name):
        config_filename = pUtil.get_model_config_filepath(model_name)
        if not config_filename.exists():
            return None
        
        meta_data = tables.get_meta_data(model_name)
        
        with open(config_filename, 'r') as config_file:
            config_data = json.load(config_file)
        
        training_config = config_data.get('training_config', {})
        block_size = training_config.get('block_size', np.nan)
        context_events = training_config.get('context_events', np.nan)
        
        if np.isnan(block_size):
            block_size = context_events * meta_data.max_sequence_length
        
        # Get the number of trainable parameters. Even though we use the log file this is technically defined by the configuration.
        num_params = np.nan
        training_log_filename = pUtil.get_training_dir(model_name) / "train_log_1.jsonl"
        with open(training_log_filename) as training_log_file:
            for jline in training_log_file:
                jdata = json.loads(jline)
                if jdata.get("message") == "Model info" and "num_params" in jdata:
                    num_params = jdata['num_params']
        
        return SimpleNamespace(
            batch_size              = training_config.get('batch_size', np.nan),
            block_size              = block_size,
            learning_rate           = training_config.get('learning_rate', np.nan),
            min_lr                  = training_config.get('min_lr', np.nan),
            lr_decay_iters          = training_config.get('lr_decay_iters', np.nan),
            lr_scheduler            = training_config.get('lr_scheduler', 'cosine_annealing_with_warmup'),
            n_layer                 = training_config.get('n_layer', np.nan),
            n_head                  = training_config.get('n_head', np.nan),
            n_embd                  = training_config.get('n_embd', np.nan),
            scheme                  = training_config.get('scheme', 'unknown'),
            preparation_name        = training_config.get('preparation_name', 'unknown'),
            num_params              = num_params
        )

    @staticmethod
    def get_training_run_data(model_name, iterations_per_epoch=-1):
        training_log_filename = pUtil.get_training_dir(model_name) / "train_log_1.jsonl"
        
        running_data, checkpointed_data = [], []
        with open(training_log_filename) as training_log_file:
            for jline in training_log_file:
                jdata = json.loads(jline)
                if jdata.get("message") == "Training progress" and "iter" in jdata:
                    current_epochs_trained = 0 if jdata['iter'] == 0 else (jdata['iter'] / iterations_per_epoch)
                    running_data.append({'iter': jdata["iter"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
                elif jdata.get("message") == "Training progress: checking checkpoint conditions":
                    current_epochs_trained = 0 if jdata['step'] == 0 else (jdata['step'] / iterations_per_epoch)
                    checkpointed_data.append({'iter': jdata["step"], 'epoch': current_epochs_trained, 'train_loss': jdata["train_loss"], 'val_loss': jdata["val_loss"]})
        
        return SimpleNamespace(
            running_df = pd.DataFrame(running_data),
            checkpointed_df = pd.DataFrame(checkpointed_data)
        )
    
    @staticmethod
    def get_metrics(model_name):
        ret_dict = SimpleNamespace(**{
            'coverage': np.nan,
            'mmd': np.nan,
            'kpd_median': np.nan,
            'kpd_error': np.nan,
            'fpd_value': np.nan,
            'fpd_error': np.nan,
            'w1m_score': np.nan,
            'w1m_score_std': np.nan,
            'w1p_avg_eta': np.nan,
            'w1p_avg_phi': np.nan,
            'w1p_avg_pt': np.nan,
            'w1p_avg_eta_std': np.nan,
            'w1p_avg_phi_std': np.nan,
            'w1p_avg_pt_std': np.nan,
        })
        
        latest_sampling_dir = pUtil.get_latest_sampling_dir(model_name)
        metrics_results_filename = latest_sampling_dir / "metrics_results.json"
        
        if not latest_sampling_dir.exists():
            return ret_dict
        if not metrics_results_filename.exists():
            return ret_dict
        
        with open(metrics_results_filename, 'r') as metrics_file:
            metrics_data = json.load(metrics_file)
        
        return SimpleNamespace(**metrics_data)
    
    # Returns all important data for a model in a dictionary
    @staticmethod
    def get_all_data(model_name):
        meta_data = tables.get_meta_data(model_name)
        config_data = tables.get_config_data(model_name)
        metrics_data = tables.get_metrics(model_name)

        iterations_per_epoch = meta_data.num_train_tokens // (config_data.batch_size * config_data.block_size)
        
        # Training information
        training_run_data = tables.get_training_run_data(model_name, iterations_per_epoch)
        iters_trained = training_run_data.running_df['iter'].max()
        min_saved_val_loss_row = training_run_data.checkpointed_df.loc[training_run_data.checkpointed_df['val_loss'].idxmin()]
        
        training_run_data = SimpleNamespace(
            iters_trained           = iters_trained,
            iters_saved             = int(min_saved_val_loss_row['iter']),
            min_saved_train_loss    = min_saved_val_loss_row['train_loss'],
            min_saved_val_loss      = min_saved_val_loss_row['val_loss'],
            running_df              = training_run_data.running_df,
            checkpointed_df         = training_run_data.checkpointed_df
        )
        
        return SimpleNamespace(**{'model_name': model_name, **vars(meta_data), **vars(config_data), **vars(metrics_data), **vars(training_run_data)})

    # Returns a DataFrame with all the important data for all models
    @staticmethod
    def get_default_df(model_names):
        columns = tables.model_all_columns
        model_data_list = [row for name in model_names if (row := vars(tables.get_all_data(name))) is not None]
        model_data_df = pd.DataFrame(model_data_list, columns=columns)
        return model_data_df