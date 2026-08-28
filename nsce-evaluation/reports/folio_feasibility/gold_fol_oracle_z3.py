"""Gold-FOL oracle experiment for the 18-example FOLIO feasibility sample.

Each group below transcribes the dataset's own `premises-FOL` field into z3
(uninterpreted Object sort, unary/binary Bool-valued predicates, explicit
existential Skolem witnesses only where the encoding needs a placeholder
constant to state a formula -- z3 handles ∃/∀ natively so no witnesses are
invented that aren't already implied by the gold formulas).

For each conclusion:
  unsat(premises AND NOT concl)  -> entailed  -> oracle label True
  unsat(premises AND concl)      -> refuted   -> oracle label False
  else                            -> oracle label Uncertain
"""
from z3 import (
    DeclareSort, Function, BoolSort, Const, Consts, ForAll, Exists,
    And, Or, Not, Implies, Xor, Solver, sat, unsat,
)

Object = DeclareSort("Object")

results = []


def oracle_label(solver, premises, concl):
    s1 = Solver(); s1.add(premises); s1.add(Not(concl))
    r1 = s1.check()
    s2 = Solver(); s2.add(premises); s2.add(concl)
    r2 = s2.check()
    if r1 == unsat:
        return "True"
    if r2 == unsat:
        return "False"
    return "Uncertain"


def check(idx, gold, premises, concl):
    got = oracle_label(None, premises, concl)
    results.append((idx, gold, got, gold == got))


# --- Group A: Zelda (idx 12,13,14) -----------------------------------------
zelda, fifa22, onemillion = Consts("zelda fifa22 onemillion", Object)
Japanese = Function("Japanese", Object, BoolSort())
VGCompany = Function("VGCompany", Object, BoolSort())
Game = Function("Game", Object, BoolSort())
Created = Function("Created", Object, Object, BoolSort())
Top10 = Function("Top10", Object, BoolSort())
SellsMoreThan = Function("SellsMoreThan", Object, Object, BoolSort())
x, y = Consts("x y", Object)

p_zelda = [
    Exists([x], And(Japanese(x), VGCompany(x), Created(x, zelda))),
    Game(zelda),  # ground fact written awkwardly inside the existential in the source text
    ForAll([x, y], Implies(And(Game(x), Top10(x), Created(x, y)), Japanese(y))),
    ForAll([x], Implies(And(Game(x), SellsMoreThan(x, onemillion)), Top10(x))),
    SellsMoreThan(zelda, onemillion),
]
check(12, "True", p_zelda, Top10(zelda))
check(13, "Uncertain", p_zelda, ForAll([x], Implies(Created(fifa22, x), And(Japanese(x), VGCompany(x)))))
check(14, "False", p_zelda, Not(Top10(zelda)))

# --- Group B: Beijing (idx 24,25,26) ----------------------------------------
beijing, s08o, s08p, so, wo, sp, wp = Consts("beijing s08o s08p so wo sp wp", Object)
Hosted = Function("Hosted", Object, Object, BoolSort())
InNorth = Function("InNorth", Object, BoolSort())
InSouth = Function("InSouth", Object, BoolSort())
SecondLargest = Function("SecondLargest", Object, BoolSort())

p_beijing = [
    InNorth(beijing),
    Hosted(beijing, s08o), Hosted(beijing, s08p),
    Hosted(beijing, so), Hosted(beijing, wo), Hosted(beijing, sp), Hosted(beijing, wp),
]
check(24, "True", p_beijing, And(Hosted(beijing, s08o), Hosted(beijing, wo)))
check(25, "Uncertain", p_beijing, InSouth(beijing))
check(26, "Uncertain", p_beijing, SecondLargest(beijing))

# --- Group C: Montana/Pierre (idx 58,59) ------------------------------------
billings, montana, butte, helena, missoula, wss, pierre, bismarck = Consts(
    "billings montana butte helena missoula wss pierre bismarck", Object
)
CityIn = Function("CityIn", Object, Object, BoolSort())
p_montana = [
    CityIn(billings, montana),
    CityIn(butte, montana), CityIn(helena, montana), CityIn(missoula, montana),
    Exists([x], And(CityIn(wss, x), CityIn(butte, x))),
    Not(CityIn(pierre, montana)),
    ForAll([x], Implies(CityIn(butte, x), Not(CityIn(pierre, x)))),
]
check(58, "False", p_montana, Exists([x], And(CityIn(butte, x), CityIn(pierre, x))))
check(59, "Uncertain", p_montana, Exists([x], And(CityIn(pierre, x), CityIn(bismarck, x))))

# --- Group D: Badults (idx 69,70) -------------------------------------------
badults, pappys, bbctwo, bbcthree, tsds, andrewcollins = Consts(
    "badults pappys bbctwo bbcthree tsds andrewcollins", Object
)
MemberOf = Function("MemberOf", Object, Object, BoolSort())
Series = Function("Series", Object, BoolSort())
Starring = Function("Starring", Object, Object, BoolSort())
PilotedOn = Function("PilotedOn", Object, Object, BoolSort())
WorkingTitle = Function("WorkingTitle", Object, Object, BoolSort())
ScriptEditorFor = Function("ScriptEditorFor", Object, Object, BoolSort())

p_badults = [
    Exists([x], And(MemberOf(x, pappys), Starring(badults, x))),
    Series(badults),
    PilotedOn(badults, bbcthree),
    WorkingTitle(tsds, badults),
    ScriptEditorFor(andrewcollins, badults),
]
check(69, "True", p_badults, Exists([x], And(ScriptEditorFor(andrewcollins, x), Series(x), WorkingTitle(tsds, x))))
check(70, "False", p_badults, ForAll([x, y], Implies(And(MemberOf(x, pappys), Starring(y, x)),
                                                       Not(Or(PilotedOn(y, bbctwo), PilotedOn(y, bbcthree))))))

# --- Group E: Space/Vacuum (idx 135,136,137) --------------------------------
space, duster = Consts("space duster", Object)
Plunger = Function("Plunger", Object, BoolSort())
Vacuum = Function("Vacuum", Object, BoolSort())
Vampire = Function("Vampire", Object, BoolSort())
Suck = Function("Suck", Object, BoolSort())
HouseHoldApp = Function("HouseHoldApp", Object, BoolSort())

p_space = [
    ForAll([x], Implies(Plunger(x), Suck(x))),
    ForAll([x], Implies(Vacuum(x), Suck(x))),
    ForAll([x], Implies(Vampire(x), Suck(x))),
    Vacuum(space),
    And(HouseHoldApp(duster), Not(Suck(duster))),
]
check(135, "True", p_space, Suck(space))
check(136, "Uncertain", p_space, Vampire(space))
check(137, "False", p_space, ForAll([x], Implies(HouseHoldApp(x), Suck(x))))

# --- Group F: ML algorithms (idx 138,139) -----------------------------------
sup, unsup, reinf = Consts("sup unsup reinf", Object)
MLAlgorithm = Function("MLAlgorithm", Object, BoolSort())
RequireLabelledData = Function("RequireLabelledData", Object, BoolSort())
TrainTextSumModel = Function("TrainTextSumModel", Object, BoolSort())

p_ml = [
    Or(MLAlgorithm(sup), MLAlgorithm(unsup), MLAlgorithm(reinf)),
    Not(RequireLabelledData(unsup)),
    Or(TrainTextSumModel(sup), TrainTextSumModel(unsup), TrainTextSumModel(reinf)),
    Not(TrainTextSumModel(reinf)),
    ForAll([x], Implies(And(MLAlgorithm(x), TrainTextSumModel(x)), RequireLabelledData(x))),
]
check(138, "True", p_ml, TrainTextSumModel(sup))
check(139, "False", p_ml, TrainTextSumModel(unsup))

# --- Group G: Heck cattle (idx 143,144) -------------------------------------
HeckCattle = Function("HeckCattle", Object, BoolSort())
Aurochs = Function("Aurochs", Object, BoolSort())
BreedingBack = Function("BreedingBack", Object, Object, BoolSort())
ArtificiallySelect = Function("ArtificiallySelect", Object, BoolSort())
Animal = Function("Animal", Object, BoolSort())
Extinct = Function("Extinct", Object, BoolSort())

p_heck = [
    ForAll([x, y], Implies(BreedingBack(x, y), ArtificiallySelect(x))),
    Exists([x, y], And(HeckCattle(x), Aurochs(y), BreedingBack(x, y))),
    ForAll([x], Implies(HeckCattle(x), Animal(x))),
    ForAll([x], Implies(Aurochs(x), Animal(x))),
    Exists([x, y], And(Animal(x), Animal(y), BreedingBack(x, y), Extinct(y))),
]
check(143, "True", p_heck, Exists([x], And(HeckCattle(x), ArtificiallySelect(x))))
check(144, "Uncertain", p_heck, ForAll([x], Implies(Aurochs(x), Extinct(x))))

# --- Group H: Remy (idx 154) ------------------------------------------------
remy = Const("remy", Object)
Desktop = Function("Desktop", Object, BoolSort())
WorkHome = Function("WorkHome", Object, BoolSort())
HomeOffice = Function("HomeOffice", Object, BoolSort())
Tidying = Function("Tidying", Object, BoolSort())
OfficeStore = Function("OfficeStore", Object, BoolSort())
Late = Function("Late", Object, BoolSort())

p_remy = [
    ForAll([x], Implies(Desktop(x), WorkHome(x))),
    ForAll([x], Implies(HomeOffice(x), Desktop(x))),
    ForAll([x], Implies(WorkHome(x), Tidying(x))),
    ForAll([x], Implies(WorkHome(x), OfficeStore(x))),
    ForAll([x], Implies(Tidying(x), Not(Late(x)))),
    Xor(And(Late(remy), Tidying(remy)), And(Not(Late(remy)), Not(Tidying(remy)))),
]
check(154, "False", p_remy, And(HomeOffice(remy), OfficeStore(remy)))

# --- Report ------------------------------------------------------------------
correct = sum(1 for r in results if r[3])
print(f"{correct}/{len(results)} oracle predictions matched gold label\n")
for idx, gold, got, ok in results:
    print(f"idx={idx:>4}  gold={gold:<10} oracle={got:<10} {'OK' if ok else 'MISMATCH'}")
