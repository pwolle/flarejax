import beartype

__all__ = ["typecheck"]


typecheck = beartype.beartype(
    conf=beartype.BeartypeConf(
        violation_type=UserWarning,
    )
)
