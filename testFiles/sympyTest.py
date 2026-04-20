import sympy as sp
import numpy as np

def CCEFunc(x,t,N,w,L_n,tau):
    return (
        (N/2)
        * (  
            2*sp.cosh(f(x,L_n,w))
            - sp.exp(f(x,L_n,w)) * g(x,L_n,w,tau,t,1)
            - sp.exp(-f(x,L_n,w)) * g(x,L_n,w,tau,t,-1)
            )
        )

def f(x,L_n,w):
    return (x-w)/L_n

def g(x,L_n,w,tau,t,const):
    return sp.erf(f(x,L_n*2,w)*sp.sqrt(tau/t) + sp.sqrt(t/tau) * const)

def landauFunc(x, c, mu):
    t = sp.symbols("t")
    return (1/(sp.pi*c))*sp.integrate(sp.exp(-t)*sp.cos((x-mu)*t/c + 2*t/sp.pi * sp.ln(t/c)), (t, 0, sp.oo))

x, t, N, L_n, D_n, w_2, tau_n, x0 = sp.symbols("x t N L_n D_n w_2 \\tau_n x0", real=True, positive=True)
T = sp.symbols("T", real=True, positive=True)
T = 0.38
L_n = sp.sqrt(D_n*tau_n)
print(1)
expr = CCEFunc(x0,t,1,w_2,L_n,tau_n)
print(2)
print(sp.latex(expr))
print(3)
x, c, t, mu = sp.symbols("x c t \\mu")
landau = landauFunc(x, 1, 0)
print(4)

f = landau
g = expr

# ---------------------------------------------------------------
# 4. Verify f is a valid PDF (optional but recommended)
# ---------------------------------------------------------------
print("Verifying f integrates to 1 over x...")
norm = sp.integrate(f, (x, -sp.oo, sp.oo))
print(f"  ∫f dx = {sp.simplify(norm)}\n")

# ---------------------------------------------------------------
# 5. Find the valid domain of x
#    A solution exists only when g(x0) = T/f(x) is achievable,
#    i.e. T/f(x) must lie within the range of g.
#    Equivalently: f(x) >= T / sup(g)
# ---------------------------------------------------------------
print("─" * 60)
print("Finding valid domain of x (where a solution exists)...")

# Supremum of g over x0 (may be oo if g is unbounded)
g_sup = sp.limit(g, x0, -sp.oo)   # adjust direction for your g
g_sup = sp.simplify(g_sup)
if g_sup == sp.oo:
    print("  g is unbounded — a solution always exists for any x.")
    x_valid_condition = sp.true
else:
    # Condition: f(x) >= T / g_sup
    threshold_expr = T / g_sup
    x_valid_condition = sp.Ge(f, threshold_expr)
    print(f"  sup(g) = {g_sup}")
    print(f"  Requires f(x) >= {sp.simplify(threshold_expr)}")

    # Solve for the boundary x values symbolically
    boundary = sp.solve(sp.Eq(f, threshold_expr), x)
    if boundary:
        print(f"  Boundary x values: {[sp.simplify(b) for b in boundary]}")
    else:
        print("  Could not find boundary symbolically — use numerical check.")

# ---------------------------------------------------------------
# 6. Solve p = T  =>  f(x)*g(x0) = T  =>  g(x0) = T/f(x)
# ---------------------------------------------------------------
rhs = T / f  # g(x0) = T / f(x)

print("\nSolving g(x0) = T / f(x) for x0...")
solutions = sp.solve(sp.Eq(g, rhs), x0)

if not solutions:
    print("  No closed-form solution found. Try sp.nsolve for numerical solution.")
else:
    print(f"  Found {len(solutions)} solution(s):\n")
    for i, sol in enumerate(solutions):
        sol_simplified = sp.simplify(sol)
        print(f"  x0_{i+1} = {sol_simplified}\n")

# ---------------------------------------------------------------
# 7. Derive the unnormalised distribution of x0
# ---------------------------------------------------------------
if solutions:
    x0_expr = solutions[0]
    x0_simplified = sp.simplify(x0_expr)

    print("─" * 60)
    print("x0 as a function of x (only valid on the domain above):")
    print(f"  x0(x) = {x0_simplified}\n")

    dx0_dx = sp.diff(x0_simplified, x)
    print(f"  dx0/dx = {sp.simplify(dx0_dx)}\n")

    # Change-of-variables to get the unnormalised PDF of x0
    print("─" * 60)
    print("Deriving unnormalised induced PDF of x0...")
    print("(Unnormalised because x values with no solution are discarded)\n")

    x_from_x0 = sp.solve(sp.Eq(x0_simplified, x0), x)

    if x_from_x0:
        induced_pdfs = []
        for j, x_inv in enumerate(x_from_x0):
            x_inv_s = sp.simplify(x_inv)
            print(f"  Branch {j+1}: x = {x_inv_s}")

            # Jacobian |dx/dx0|
            jacobian = sp.Abs(sp.diff(x_inv, x0))
            jacobian = sp.simplify(jacobian)

            # Check domain of this branch (x_inv must be real)
            # The expression under any sqrt must be >= 0
            inner_expr = None
            for atom in x_inv_s.atoms(sp.sqrt):
                inner_expr = atom.args[0]

            if inner_expr is not None:
                domain_condition = sp.Ge(inner_expr, 0)
                domain_x0 = sp.solve(domain_condition, x0)
                print(f"  Valid x0 domain: {inner_expr} >= 0  =>  {domain_x0}")
            else:
                print(f"  x0 domain: all real (no sqrt constraint)")

            f_sub = f.subs(x, x_inv)
            induced_pdf = sp.simplify(f_sub * jacobian)
            induced_pdfs.append(induced_pdf)
            print(f"  Unnormalised p(x0) = {induced_pdf}\n")

        # Total unnormalised mass (probability that a solution exists)
        print("─" * 60)
        print("Computing total mass Z = P(solution exists)...")
        try:
            total_mass = sum(
                sp.integrate(pdf, (x0, -sp.oo, sp.oo))
                for pdf in induced_pdfs
            )
            total_mass = sp.simplify(total_mass)
            print(f"  Z = {total_mass}")
            print(f"  (Z < 1 confirms the distribution is unnormalised as expected)")
        except Exception as e:
            print(f"  Symbolic integration failed: {e}")
            print("  Use numerical integration to compute Z.")
    else:
        print("  Could not invert x0(x) analytically.")

# ---------------------------------------------------------------
# 8. Numerical evaluation with validity check
# ---------------------------------------------------------------
print("\n" + "─" * 60)
print("Numerical example (mu=0, sigma=1, lambda=1, T=0.1):")

if solutions:
    params = {mu: 0, sigma: 1, lam: 1, T: 0.1}
    x0_num = x0_simplified.subs(params)
    f_num = f.subs(params)
    print(f"  x0(x) = {x0_num}\n")

    x0_fn = sp.lambdify(x, x0_num, modules='numpy')
    f_fn  = sp.lambdify(x, f_num,  modules='numpy')

    x_vals = np.linspace(-4, 4, 9)
    print(f"  {'x':>6}  {'f(x)':>10}  {'x0':>10}  {'valid?':>8}")
    print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*8}")
    for xv in x_vals:
        fx = float(f_fn(xv))
        try:
            # Suppress numpy's sqrt-of-negative warning: we handle it explicitly
            with np.errstate(invalid='ignore'):
                raw = x0_fn(xv)
            is_complex = np.iscomplex(raw) and abs(np.imag(complex(raw))) > 1e-10
            x0v = float(np.real(raw))
            valid = not is_complex and np.isfinite(x0v)
        except Exception:
            x0v, valid = float('nan'), False

        flag = "yes" if valid else "DISCARDED"
        x0_str = f"{x0v:>10.4f}" if valid else f"{'—':>10}"
        print(f"  {xv:>6.2f}  {fx:>10.6f}  {x0_str}  {flag:>8}")