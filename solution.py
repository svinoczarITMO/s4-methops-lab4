import sympy as sp

f = lambda x, y: x**2 - x*y + y**2 - 2*x + y 
f_dx = lambda x, y: 2*x - y - 2
f_dy = lambda x, y: -x + 2*y + 1
gradient = lambda x, y: [f_dx(x, y), f_dy(x, y)]


epsilon = 0.05
lmd = 0.25
x0, y0 = 0, 0

def gradient_descent(x_0, y_0):
    x = x_0
    y = y_0
    grad = gradient(x, y)
    x -= lmd * grad[0]
    y -= lmd * grad[1]
    f_new = f(x, y)
    if (f_new - f(x, y)) < epsilon:
        return x, y, f(x, y)
    else: return gradient_descent(x, y)


def rapid_descent(x_0, y_0):
    x, y = x_0, y_0
    dx, dy = f_dx(x, y), f_dy(x, y)
    
    h = sp.Symbol('h')
    func = f(x-dx*h, y-dy*h)
    derivative = sp.diff(func, h)
    solution = sp.solve(derivative, h)
    h_result = solution[0]

    x1, y1 = x - h_result*dx, y - h_result*dy
    
    if (f_dx(x1, y1)**2 + f_dy(x1, y1)**2)**0.5 <= epsilon:
        return (sp.N(x1), sp.N(y1), sp.N(f(x1, y1)))
    else: return(rapid_descent(x1, y1))



grad_x, grad_y, grad_z = gradient_descent(x0, y0)
rapid_x, rapid_y, rapid_z = rapid_descent(x0, y0)

print("ГРАДИЕНТНЫЙ СПУСК:")
print(f"Минимум функции f(x, y) достигается в точке ({grad_x}, {grad_y}).")
print(f"Значение функции в этой точке: {grad_z}")

print("\n")

print("НАИСКОРЕЙШИЙ СПУСК:")
print(f"Минимум функции f(x, y) достигается в точке ({rapid_x}, {rapid_y}).")
print(f"Значение функции в этой точке: {rapid_z}")