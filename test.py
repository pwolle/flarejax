import flarejax_strict as flarejax


nested = {
    "a": {
        "b": {
            "c": 1,
        },
    },
    "d": [1, 2, 3, {"e": 4}],
}


nested = flarejax.ConfigMapping(nested)
print(nested)
