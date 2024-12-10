
# Code stolen from: https://github.com/mCodingLLC/VideosSampleCode/tree/master/videos/135_modern_logging

import json
import logging
import logging.config
import logging.handlers
import pathlib
import datetime as dt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))

LOG_RECORD_BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
    "taskName",
}

class pLoggingJSONFormatter(logging.Formatter):
    def __init__(self, *, fmt_keys: dict[str, str] | None = None):
        super().__init__()
        self.fmt_keys = fmt_keys if fmt_keys is not None else {}

    def format(self, record: logging.LogRecord) -> str:
        message = self._prepare_log_dict(record)
        return json.dumps(message, default=str)

    def _prepare_log_dict(self, record: logging.LogRecord):
        always_fields = {
            "message": record.getMessage(),
            "timestamp": dt.datetime.fromtimestamp(record.created, tz=dt.timezone.utc).isoformat(),
        }
        
        if record.exc_info is not None:
            always_fields["exc_info"] = self.formatException(record.exc_info)

        if record.stack_info is not None:
            always_fields["stack_info"] = self.formatStack(record.stack_info)

        message = {
            key: msg_val
            if (msg_val := always_fields.pop(val, None)) is not None
            else getattr(record, val)
            for key, val in self.fmt_keys.items()
        }
        
        message.update(always_fields)
        
        for key, val in record.__dict__.items():
            if key not in LOG_RECORD_BUILTIN_ATTRS:
                message[key] = val

        return message

class NonErrorFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool | logging.LogRecord:
        return record.levelno <= logging.INFO

existing_loggers = []

ALL_OUTPUT_LEVEL = 25
FILE_ONLY_LEVEL = 15

def all_output(self, message, *args, **kwargs):
    if self.isEnabledFor(ALL_OUTPUT_LEVEL):
        self._log(ALL_OUTPUT_LEVEL, message, args, **kwargs)
        
def file_only(self, message, *args, **kwargs):
    if self.isEnabledFor(FILE_ONLY_LEVEL):
        self._log(FILE_ONLY_LEVEL, message, args, **kwargs)
        
def update_config(log_filename):
    config = json.load(open(pathlib.Path(os.path.join(script_dir, "log_config.json"))))
    
    # Update all handlers to have absolute paths
    for handler_name, handler in config['handlers'].items():
        if 'filename' in handler:
            handler['filename'] = os.path.join(script_dir, "logs/" + log_filename if log_filename != None else handler['filename'])
    
    logging.config.dictConfig(config)
        
def setup_logging():
    logging.addLevelName(ALL_OUTPUT_LEVEL, "AllOutput")
    logging.addLevelName(FILE_ONLY_LEVEL, "FileOnly")
    
    logging.Logger.all_output = all_output
    logging.Logger.file_only = file_only
    
    update_config(None)
    logging.basicConfig(level="FileOnly")
    
def create_logger(phase):
    now = dt.datetime.now().strftime("%m-%d-%Y-%H-%M-%S")
    logger_name = phase + '_' + now
    new_logger = logging.getLogger(logger_name)
    existing_loggers.append(new_logger)
    update_config(logger_name + '.jsonl')
    return len(existing_loggers) - 1

def info(logger_index, message, extra=None):
    if extra is None:
        existing_loggers[logger_index].all_output(message, extra=extra)
    else:
        existing_loggers[logger_index].file_only(message, extra=extra)

setup_logging()