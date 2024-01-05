properties = {}


def init():
    if len(properties) == 0:
        f = open("ic.properties", "r")
        for line in f:
            line = line.strip()
            if len(line) > 0:
                tokens = line.split("=")
                key = tokens[0]
                val = tokens[1]
                properties[key] = val
        print(properties)
        f.close()

def get_val(key):
    init()
    return properties[key]