import os
import csv
from datetime import datetime

# --------------------------------
# logger
# --------------------------------
class LOG(object):
    def __init__(self, filepath, filename, field_name):
        self.filepath = filepath
        self.filename = filename
        self.field_name = field_name

        self.logfile, self.logwriter = csv_log(file_name=os.path.join(filepath, filename+'.csv'), field_name=field_name)
        self.logwriter.writeheader()

    def record(self, *args):
        dict = {}
        for i in range(len(self.field_name)):
            dict[self.field_name[i]]=args[i]
        self.logwriter.writerow(dict)

    def close(self):
        self.logfile.close()

    def print(self, msg):
        logT(msg)

def csv_log(file_name, field_name):
    assert file_name is not None
    assert field_name is not None
    logfile = open(file_name, 'w')
    logwriter = csv.DictWriter(logfile, fieldnames=field_name)
    return logfile, logwriter

def logT(*args, **kwargs):
     print(get_timestamp(), *args, **kwargs)

def get_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H:%M:%S')