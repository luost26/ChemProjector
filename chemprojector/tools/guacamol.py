import functools

from guacamol import goal_directed_benchmark, scoring_function, standard_benchmarks

from chemprojector.chem.mol import Molecule


class GuacamolScoringFunction:
    def __init__(self, name: str, objective: scoring_function.ScoringFunction):
        super().__init__()
        self._name = name
        self._objective = objective

    @classmethod
    def from_benchmark(cls, benchmark: goal_directed_benchmark.GoalDirectedBenchmark):
        return GuacamolScoringFunction(benchmark.name, benchmark.objective)

    def __call__(self, mol: Molecule | str) -> float:
        if isinstance(mol, str):
            return self._objective.score(mol)
        else:
            return self._objective.score(mol.csmiles)

    @classmethod
    @functools.cache
    def get_preset(cls, name: str) -> "GuacamolScoringFunction":
        return getattr(cls, name.replace(" ", "_"))()

    @classmethod
    def Zaleplon_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.zaleplon_with_other_formula())

    @classmethod
    def Sitagliptin_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.sitagliptin_replacement())

    @classmethod
    def Valsartan_SMARTS(cls):
        return cls.from_benchmark(standard_benchmarks.valsartan_smarts())

    @classmethod
    def Scaffold_Hop(cls):
        return cls.from_benchmark(standard_benchmarks.scaffold_hop())

    @classmethod
    def Fexofenadine_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.hard_fexofenadine())

    @classmethod
    def Osimertinib_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.hard_osimertinib())

    @classmethod
    def Deco_Hop(cls):
        return cls.from_benchmark(standard_benchmarks.decoration_hop())

    @classmethod
    def Amlodipine_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.amlodipine_rings())

    @classmethod
    def Ranolazine_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.ranolazine_mpo())

    @classmethod
    def Perindopril_MPO(cls):
        return cls.from_benchmark(standard_benchmarks.perindopril_rings())
