import dataclasses
import itertools
import random
from collections.abc import Iterable
from typing import TypeAlias

import numpy as np

from .matrix import ReactantReactionMatrix
from .mol import Molecule
from .reaction import Reaction

_NumReactants: TypeAlias = int
_MolOrRxnIndex: TypeAlias = int
_TokenType: TypeAlias = tuple[_NumReactants, _MolOrRxnIndex]


def _flatten(l):
    for el in l:
        if isinstance(el, list):
            yield from _flatten(el)
        else:
            yield el


@dataclasses.dataclass
class _Node:
    mol: Molecule
    rxn: Reaction | None
    token: _TokenType
    children: list["_Node"]

    def to_str(self, depth: int) -> str:
        pad = " " * depth * 2
        lines = [f"{pad}{self.mol.smiles}"]
        if self.rxn is not None:
            for c in self.children:
                lines.append(f"{c.to_str(depth + 1)}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"Node(\n{self.to_str(1)}\n)"


class Stack:
    def __init__(self) -> None:
        super().__init__()
        self._mols: list[Molecule] = []
        self._rxns: list[Reaction | None] = []
        self._tokens: list[_TokenType] = []
        self._stack: list[set[Molecule]] = []

    @property
    def mols(self) -> tuple[Molecule, ...]:
        return tuple(self._mols)

    @property
    def rxns(self) -> tuple[Reaction | None, ...]:
        return tuple(self._rxns)

    @property
    def tokens(self) -> tuple[_TokenType, ...]:
        return tuple(self._tokens)

    def get_top(self) -> set[Molecule]:
        return self._stack[-1]

    def get_second_top(self) -> set[Molecule]:
        return self._stack[-2]

    def push_mol(self, mol: Molecule, index: int) -> None:
        self._mols.append(mol)
        self._rxns.append(None)
        self._tokens.append((0, index))
        self._stack.append({mol})

    def push_rxn(self, rxn: Reaction, index: int, product_limit: int | None = None) -> bool:
        if len(self._stack) < rxn.num_reactants:
            return False

        prods: list[Molecule] = []
        if rxn.num_reactants == 1:
            for r in self.get_top():
                prods += rxn([r])
        elif rxn.num_reactants == 2:
            for r1, r2 in itertools.product(self.get_top(), self.get_second_top()):
                if product_limit is not None and len(prods) >= product_limit:
                    break
                prods += rxn([r1, r2]) + rxn([r2, r1])
        else:
            return False

        if len(prods) == 0:
            return False
        if product_limit is not None:
            prods = prods[:product_limit]
        prod: Molecule = random.choice(prods)

        self._mols.append(prod)
        self._rxns.append(rxn)
        self._tokens.append((rxn.num_reactants, index))
        for _ in range(rxn.num_reactants):
            self._stack.pop()
        self._stack.append(set(prods))
        return True

    def get_tree(self) -> _Node:
        stack: list[_Node] = []
        for i in range(len(self._tokens)):
            token = self._tokens[i]
            n_react = token[0]
            if n_react > 0:
                item = _Node(self._mols[i], self._rxns[i], token, [])
                for _ in range(n_react):
                    item.children.append(stack.pop())
                stack.append(item)
            else:
                stack.append(_Node(self._mols[i], self._rxns[i], token, []))
        return stack[-1]

    def get_postfix_tokens(self) -> tuple[_TokenType, ...]:
        return tuple(self._tokens)

    def __len__(self) -> int:
        return len(self._mols)

    def __getitem__(self, index: int) -> Molecule:
        return self._mols[index]

    def get_mol_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] == 0 else None for t in self.tokens]

    def get_rxn_idx_seq(self) -> list[int | None]:
        return [t[1] if t[0] > 0 else None for t in self.tokens]

    def count_reactions(self) -> int:
        cnt = 0
        for rxn in self._rxns:
            if rxn is not None:
                cnt += 1
        return cnt

    def get_state_repr(self) -> str:
        rl: list[str] = []
        for s in self._stack:
            sl = list(map(lambda m: m.csmiles, s))
            sl.sort()
            rl.append(",".join(sl))
        return ";".join(rl)

    def get_action_string(self, delim: str = ";") -> str:
        tokens: list[str] = []
        for mol, (num_reactants, idx) in zip(self._mols, self._tokens):
            if num_reactants == 0:
                tokens.append(mol.smiles)
            else:
                tokens.append(f"R{idx}")
        return delim.join(tokens)


def create_init_stack(matrix: ReactantReactionMatrix, weighted_ratio: float = 0.0) -> Stack:
    stack = Stack()

    if weighted_ratio != 0.0:
        prob_w = matrix.reactant_count[matrix.seed_reaction_indices].astype(np.float32)
        prob_w = prob_w / prob_w.sum()
        prob_u = np.ones_like(prob_w) / len(prob_w)
        prob = weighted_ratio * prob_w + (1 - weighted_ratio) * prob_u
        rxn_index: int = np.random.choice(matrix.seed_reaction_indices, p=prob)
    else:
        rxn_index = random.choice(matrix.seed_reaction_indices)
    rxn_col = matrix.matrix[:, rxn_index]
    rxn = matrix.reactions[rxn_index]

    if rxn.num_reactants == 2:
        m1 = np.random.choice(np.bitwise_and(rxn_col, 0b01).nonzero()[0])
        m2 = np.random.choice(np.bitwise_and(rxn_col, 0b10).nonzero()[0])
        if random.randint(0, 1) % 2 == 1:
            m1, m2 = m2, m1
        stack.push_mol(matrix.reactants[m1], m1)
        stack.push_mol(matrix.reactants[m2], m2)
    elif rxn.num_reactants == 1:
        m = np.random.choice(rxn_col.nonzero()[0])
        stack.push_mol(matrix.reactants[m], m)

    stack.push_rxn(rxn, rxn_index)
    return stack


def expand_stack(stack: Stack, matrix: ReactantReactionMatrix):
    matches = matrix.reactions.match_reactions(random.choice(list(stack.get_top())))
    if len(matches) == 0:
        return stack, False
    rxn_index = random.choice(list(matches.keys()))
    reactant_flag = 1 << matches[rxn_index][0]

    rxn_col = matrix.matrix[:, rxn_index]
    s_indices = np.logical_and(rxn_col != 0, rxn_col != reactant_flag).nonzero()[0]
    if len(s_indices) == 0:
        return stack, False
    s_index = np.random.choice(s_indices)
    stack.push_mol(matrix.reactants[s_index], s_index)
    stack.push_rxn(matrix.reactions[rxn_index], rxn_index)
    return stack, True


def create_stack(
    matrix: ReactantReactionMatrix,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
) -> Stack:
    stack = create_init_stack(matrix, weighted_ratio=init_stack_weighted_ratio)
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if not changed:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break
    return stack


def create_stack_step_by_step(
    matrix: ReactantReactionMatrix,
    max_num_reactions: int = 5,
    max_num_atoms: int = 80,
    init_stack_weighted_ratio: float = 0.0,
) -> Iterable[Stack]:
    stack = create_init_stack(matrix, weighted_ratio=init_stack_weighted_ratio)
    yield stack
    for _ in range(1, max_num_reactions):
        stack, changed = expand_stack(stack, matrix)
        if changed:
            yield stack
        else:
            break
        if max(map(lambda m: m.num_atoms, stack.get_top())) > max_num_atoms:
            break
