from logic import *

AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
knowledge0 = And(
    Or(
        # If A is a knight
        And(AKnight, And(AKnight, AKnave)),
        # If A is a knave
        And(AKnave, Not(And(AKnight, AKnave)))
    ),
    Or(
        And(AKnave, Not(AKnight)),
        And(Not(AKnave), AKnight)
    )
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
knowledge1 = And(
    Or(
        # If A is a knight
        And(AKnight, And(AKnave, BKnave)),
        # If A is a knave
        And(AKnave, Not(And(AKnave, BKnave)))
    ),
    Or(
        And(AKnave, Not(AKnight)),
        And(Not(AKnave), AKnight)
    ),
    Or(
        And(BKnave, Not(BKnight)),
        And(Not(BKnave), BKnight)
    )
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
knowledge2 = And(
    Or(
        # If A is a knight
        And(AKnight, Or(And(AKnave, BKnave), And(AKnight, BKnight))),
        # If A is a knave
        And(AKnave, Not(Or(And(AKnave, BKnave), And(AKnight, BKnight))))
    ),
    Or(
        # If B is a knight
        And(BKnight, Or(And(AKnave, BKnight), And(AKnight, BKnave))),
        # If B is a knave
        And(BKnave, Not(Or(And(AKnave, BKnight), And(AKnight, BKnave))))
    ),
    Or(
        And(AKnave, Not(AKnight)),
        And(Not(AKnave), AKnight)
    ),
    Or(
        And(BKnave, Not(BKnight)),
        And(Not(BKnave), BKnight)
    )
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
knowledge3 = And(
    Or(  # A says either "I am a knight." or "I am a knave.", but you don't know which.
        # If A is a knight
        And(AKnight, Or(AKnave, AKnight)),
        # If A is a knave
        And(AKnave, Not(Or(AKnave, AKnight)))
    ),
    Or(  # B says "A said 'I am a knave'."
        # If B is a knight
        And(BKnight, Or(
            And(AKnight, AKnave),
            And(AKnave, Not(AKnave))
        )),
        # If B is a knave, A didnt say anything
        BKnave
    ),
    Or(  # B says "C is a knave."
        # If B is a knight
        And(BKnight, CKnave),
        # If B is a knave
        And(BKnave, Not(CKnave))
    ),
    Or(  # C says "A is a knight."
        # If C is a knight
        And(CKnight, AKnight),
        # If C is a knave
        And(CKnave, Not(AKnight))
    ),
    Or(
        And(AKnave, Not(AKnight)),
        And(Not(AKnave), AKnight)
    ),
    Or(
        And(BKnave, Not(BKnight)),
        And(Not(BKnave), BKnight)
    ),
    Or(
        And(CKnave, Not(CKnight)),
        And(Not(CKnave), CKnight)
    )

)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
