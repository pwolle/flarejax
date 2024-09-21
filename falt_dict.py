from pprint import pprint

nested_dict = {
    "key1": "value1",
    "key2": {
        "key3": "value3",
        "key4": {
            "key5": "value5",
        },
    },
}


def flatten_dict(nested_dict, path=()):
    items = []

    for key, value in nested_dict.items():
        new_path = path + (key,)

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_path).items())
            continue

        items.append((new_path, value))

    return dict(items)


def unflatten_dict(flat_dict):
    nested_dict = {}
    for path, value in flat_dict.items():
        current = nested_dict
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value

    return nested_dict


flat_dict = flatten_dict(nested_dict)
reco_dict = unflatten_dict(flat_dict)

pprint(flat_dict)
pprint(reco_dict)
