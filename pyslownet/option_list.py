from pyslownet.slownet import Metadata


def read_data_cfg(filename: str) -> list:

    pass


def get_metadata(file: str) -> Metadata:
    pass


def read_option(s: str, options: list) -> int:
    pass


def option_insert():
    pass


def option_unused():
    pass


def option_find():
    pass


def option_find_str():
    pass


def option_find_int(l: list, key: str, define: int) -> dict[str, str]:
    char *v = option_find(l, key);
    if(v) return atoi(v);
    fprintf(stderr, "%s: Using default '%d'\n", key, define);
    return def;

def option_find_int_quiet(l: list, key: str, define: int)
    char *v = option_find(l, key);
    if(v) return atoi(v);
    return def;

def option_find_float_quiet(l: list, key: str, define: float)
    char *v = option_find(l, key);
    if(v) return atof(v);
    return def;

def option_find_float(l: list, key: str, define: float)
    char *v = option_find(l, key);
    if(v) return atof(v);
    fprintf(stderr, "%s: Using default '%lf'\n", key, def);
    return def;
