All figures are produced using a dedicated plotting utility that automatically formats the histogram and ratio axes, applies consistent binning across models, and overlays statistical uncertainties. The reference distribution is displayed in black while the generative models are shown in color. Optional annotations display the number of events used for each dataset and other model metadata such as codebook size.

The resulting plots provide a compact visual comparison between the reference simulation and the generative model outputs, allowing both the shape of the distributions and their relative deviations to be inspected simultaneously.

# Figure Generation and Data Analysis

\section{Figure Generation and Data Analysis}

A central requirement for generative models in particle physics is the generated events reproduce the statistical properties of the underlying physical process. While individual generated events may vary, the ensemble of generated samples should match the distributions produced by the reference simulation.

% @TODO: I am unsure if we should include the energy conservation figures.
To evaluate this agreement, we compare distributions of physically meaningful observables extracted from both the \texttt{Geant4} reference dataset and the generated samples produced by \textit{particleGPT}. These observables include quantities such as particle multiplicities, kinematic variables, and event-level energy conservation. If the generative model has successfully learned the structure of the physical process, these distributions should agree within statistical uncertainties.

The comparison is performed using histogram-based analyses together with ratio plots to highlight deviations between generated and reference distributions. Agreement between the two distributions provides evidence the generative model captures the relevant correlations and statistical structure present in the simulated events.

The following sections describe the procedure used to extract observables from the event data and construct the comparison figures used throughout this work.

## Extraction of Observables
\subsection{Extraction of Observables}

% @TODO: add information about converting the pt, eta, phi to the four-momenta.
Before constructing distributions, the datasets are processed to extract the observable of interest. In this preprocessing step, the incident particle is removed and only the outgoing particles are retained. For an observable $x$, the resulting analysis vector is therefore
$$
x = \{x_1, x_2, \dots , x_N\},
$$
where $N$ is the total number of valid particles (or events) contributing to that observable. Depending on the quantity being studied, the analysis may operate either on all outgoing particles or only on the leading particle in each event. The leading particle is defined as the outgoing particle with the highest energy within the event.

Certain derived quantities can also be constructed from the particle four-momenta. For example, the total incoming and outgoing system energies are computed using relativistic four-vector addition,
$$
E = \sqrt{m^2 + p_x^2 + p_y^2 + p_z^2},
$$
and summed across particles to obtain event-level quantities such as the incoming and outgoing total energy.

## Histogram Construction
\subsection{Histogram Construction}

For each observable, a histogram is constructed using a common binning scheme shared by the earlier tokenization. If $x$ represents the observable being studied and the bin edges are given by
$$
\{b_0, b_1, \dots, b_{n}\},
$$
then the number of entries in bin $i$ is
$$
N_i = \#\{x \mid b_i \le x < b_{i+1}\}.
$$

These counts are displayed either as raw counts or normalized densities depending on whether it is possible to ensure equal numbers of particles for every analysis. The statistical uncertainty on each bin is assumed to follow Poisson statistics and is therefore taken as
$$
\sigma_i = \sqrt{N_i}.
$$
The reference distribution (from \texttt{Geant4}) is drawn as a step histogram with an uncertainty band, while distributions from the generative models are overlaid using colored step histograms with the same binning.

## Ratio Comparison
\subsection{Ratio Comparison}

To highlight differences between generated data and the reference simulation, each figure includes a ratio panel showing the relative deviation between the two distributions. The ratio is computed using the normalized bin densities,
$$
R_i = \frac{D_i^{\mathrm{gen}}}{D_i^{\mathrm{ref}}},
$$
where $D_i$ denotes the normalized density in bin $i$. The uncertainty on the ratio is obtained through standard error propagation,
$$
\sigma_{R_i}
=
R_i
\sqrt{
\left(\frac{\sigma_{D_i^{\mathrm{gen}}}}{D_i^{\mathrm{gen}}}\right)^2
+
\left(\frac{\sigma_{D_i^{\mathrm{ref}}}}{D_i^{\mathrm{ref}}}\right)^2
}.
$$
Reference guide lines are drawn at
$$
R = 1, \quad R = 1 \pm 0.05, \quad R = 1 \pm 0.10,
$$
to visually indicate the level of agreement between the generated and reference distributions.

## Plot Presentation
\subsection{Plot Presentation}

% @TODO: this might be too much detail?
All figures are produced using a dedicated plotting utility that automatically formats the histogram and ratio axes, applies consistent binning across models, and overlays statistical uncertainties. The reference distribution is displayed in black while the generative models are shown in color. Optional annotations display the number of events used for each dataset and other model metadata such as codebook size.

The resulting plots provide a compact visual comparison between the reference simulation and the generative model outputs, allowing both the shape of the distributions and their relative deviations to be inspected simultaneously.