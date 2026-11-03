import cplex

print("CPLEX version:", cplex.__version__)

model = cplex.Cplex()
model.set_log_stream(None)
model.set_warning_stream(None)
model.set_results_stream(None)

# Deliberately exceed the Community Edition variable limit
n = 1101

model.variables.add(
    obj=[1.0] * n,
    lb=[0.0] * n,
    ub=[1.0] * n,
)

try:
    model.solve()
    print("\nSUCCESS")
    print("A model with", n, "variables solved.")
    print("You are NOT restricted by the CPLEX Community Edition size limit.")
    print("This is consistent with an Academic/full CPLEX license.")

except cplex.exceptions.CplexSolverError as e:
    print("\nCPLEX ERROR:")
    print(e)

    if "1016" in str(e) or "Community Edition" in str(e):
        print("\nYou are using the COMMUNITY EDITION.")
    else:
        print("\nThe error was not the normal Community Edition size-limit error.")