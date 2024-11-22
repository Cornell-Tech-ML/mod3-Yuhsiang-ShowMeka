from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple, Protocol


# ## Task 1.1
# Central Difference calculation


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
    ----
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
    -------
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$

    """
    # TODO: Implement for Task 1.1.
    # Create two lists of values, one for f(x+ε) and one for f(x-ε)
    vals_plus = list(vals)
    vals_minus = list(vals)

    # Modify only the argument we're differentiating with respect to
    vals_plus[arg] += epsilon
    vals_minus[arg] -= epsilon

    # Compute f(x+ε) and f(x-ε)
    f_plus = f(*vals_plus)
    f_minus = f(*vals_minus)

    # Return the central difference
    return (f_plus - f_minus) / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        """Accumulates the derivative of the variable."""
        ...

    @property
    def unique_id(self) -> int:
        """unique_id is also used for history"""
        ...

    def is_leaf(self) -> bool:
        """Leaf means no parent"""
        ...

    def is_constant(self) -> bool:
        """Constant means no derivative"""
        ...

    @property
    def parents(self) -> Iterable["Variable"]:
        """Parents is an iterable of variables that are the parents of the current variable"""
        ...

    def chain_rule(self, d_output: Any) -> Iterable[Tuple[Variable, Any]]:
        """chain_rule is an iterable of tuples of variables and their derivatives"""
        ...


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """Computes the topological order of the computation graph.

    Args:
    ----
        variable: The right-most variable

    Returns:
    -------
        Non-constant Variables in topological order starting from the right.

    """
    # TODO: Implement for Task 1.4.
    # topological sort using dfs
    visited = set()
    sorted_nodes = []

    def visit(v: Variable) -> None:
        if v.unique_id not in visited and not v.is_constant():
            visited.add(v.unique_id)
            for parent in v.parents:
                visit(parent)
            sorted_nodes.append(v)

    visit(variable)
    return list(reversed(sorted_nodes))


def backpropagate(variable: Variable, deriv: Any) -> None:
    """Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    # TODO: Implement for Task 1.4.
    sorted_variables = topological_sort(variable)
    grad_map = {}
    grad_map[variable.unique_id] = deriv

    for var in sorted_variables:
        if not var.is_constant():
            current_grad = grad_map.get(var.unique_id, 0.0)
            if var.is_leaf():
                var.accumulate_derivative(current_grad)
            else:
                for parent, grad in var.chain_rule(current_grad):
                    if parent.unique_id not in grad_map:
                        grad_map[parent.unique_id] = 0.0
                    grad_map[parent.unique_id] += grad


@dataclass
class Context:
    """Context class is used by `Function` to store information during the forward pass."""

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        """Store the given `values` if they need to be used during backpropagation."""
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        """Returns the saved values."""
        return self.saved_values
