import beartype
from jaxtyping import jaxtyped

__all__ = ["typecheck"]


typecheck = jaxtyped(
    typechecker=beartype.beartype(
        conf=beartype.BeartypeConf(
            violation_type=UserWarning,
        )
    )
)
