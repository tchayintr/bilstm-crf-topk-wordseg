import numpy as np

import constants


class Key2Values(object):
    def __init__(self):
        self.key2values = {}

    def __len__(self):
        return len(self.key2values)

    def __str__(self):
        return str(self.key2values)

    def add(self, key, val):
        if key in self.key2values:
            vals = self.key2values[key]
        else:
            vals = set()
            self.key2values[key] = vals
        vals.add(val)

    def get(self, key):
        if key in self.key2values:
            return self.key2values[key]
        else:
            return set()

    def keys(self):
        return self.key2values.keys()


class IndexTable(object):
    def __init__(self, str2id=None, unk_symbol=None):
        self.unk_id = -1

        if str2id:
            self.str2id = str2id
        else:
            self.str2id = {}
            if unk_symbol:
                self.set_unk(unk_symbol)

        self.id2str = {}

    def set_unk(self, unk_symbol):
        if self.unk_id < 0:
            self.unk_id = len(self.str2id)
            self.str2id[unk_symbol] = self.unk_id
            return self.unk_id

        else:
            return -1

    def __len__(self):
        return len(self.str2id)

    def __entries__(self):
        return list(self.str2id.keys())

    def create_id2str(self):
        self.id2str = {v: k for k, v in self.str2id.items()}

    def get_id(self, key, update=False):
        if key in self.str2id:
            return self.str2id[key]
        elif update:
            id = np.int32(len(self.str2id))
            self.str2id[key] = id
            return id
        else:
            return self.unk_id

    def add_entries(self, strs):
        for s in strs:
            self.get_id(s, update=True)


class Dictionary(object):
    def __init__(self):
        self.tables = {}
        self.tries = {}

    def create_table(self, table_name):
        # string to index table
        self.tables[table_name] = IndexTable()

    def create_id2strs(self):
        for table in self.tables.values():
            table.create_id2str()

    def has_table(self, table_name):
        return table_name in self.tables
