import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

def turbulent_flow_model(t, y, U, L, C_mu, C_1, C_2, noise_intensity):
    """
    Sistema de ecuaciones para el modelo k-epsilon con términos estocásticos.
    y[0] = k (energía cinética turbulenta)
    y[1] = epsilon (tasa de disipación)
    """
    k, eps = y
    
    # Evitar división por cero o valores negativos
    k = max(k, 1e-10)
    eps = max(eps, 1e-10)
    
    # Término de producción de turbulencia (simplificado)
    P_k = C_mu * k**2 / eps * (U / L)**2
    
    # Ecuaciones deterministas
    dk_dt = P_k - eps
    de_dt = C_1 * (eps / k) * P_k - C_2 * (eps**2 / k)
    
    # Términos estocásticos (ruido blanco de Wiener)
    # dW/dt es ruido blanco, que se aproxima con N(0,1)/sqrt(dt)
    # Para la integración numérica, se multiplica por sqrt(dt)
    # Aquí lo modelamos como una perturbación proporcional.
    dk_dt += noise_intensity * np.sqrt(k) * np.random.normal(0, 1)
    de_dt += noise_intensity * np.sqrt(eps) * np.random.normal(0, 1)
    
    return [dk_dt, de_dt]

def simulate_flow(t_span, t_eval, y0, params):
    """Resuelve el sistema de EDOs para el flujo turbulento."""
    solution = solve_ivp(
        fun=turbulent_flow_model,
        t_span=t_span,
        y0=y0,
        t_eval=t_eval,
        args=tuple(params.values()),
        method='RK45'
    )
    return solution.t, solution.y

def plot_results(t, k, eps, nu_t):
    """Visualiza los resultados de la simulación."""
    plt.figure(figsize=(12, 9))
    plt.suptitle("Simulación Estocástica de Flujo Turbulento (Modelo k-ε)", fontsize=16)

    plt.subplot(3, 1, 1)
    plt.plot(t, k, 'b-', label='k (Energía Cinética)')
    plt.ylabel('Energía Cinética\nTurbulenta (m²/s²)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(t, eps, 'r-', label='ε (Tasa de Disipación)')
    plt.ylabel('Tasa de Disipación\n(m²/s³)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(t, nu_t, 'g-', label='νt (Viscosidad Turbulenta)')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Viscosidad\nTurbulenta (m²/s)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

if __name__ == "__main__":
    # Para reproducibilidad de la componente aleatoria
    np.random.seed(42)

    # Parámetros del flujo y del modelo
    params = {
        "U": 1.0,     # Velocidad media (m/s)
        "L": 1.0,     # Longitud característica (m)
        "C_mu": 0.09,
        "C_1": 1.44,
        "C_2": 1.92,
        "noise_intensity": 0.05 # Intensidad del ruido estocástico
    }

    # Configuración de la simulación
    t_span = [0, 20]
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    k0 = 0.01 * params["U"]**2
    e0 = params["C_mu"]**0.75 * k0**1.5 / (0.07 * params["L"])
    y0 = [k0, e0]

    # Ejecutar simulación y obtener resultados
    t, y = simulate_flow(t_span, t_eval, y0, params)
    k, eps = y
    nu_t = params["C_mu"] * k**2 / eps

    # Visualizar resultados
    plot_results(t, k, eps, nu_t)

    # Análisis estadístico de la fase estacionaria (última mitad de la simulación)
    print("Estadísticas del flujo turbulento (fase estacionaria):")
    print(f"Media de k: {np.mean(k[-len(k)//2:]):.4e} m²/s²")
    print(f"Desviación estándar de k: {np.std(k[-len(k)//2:]):.4e} m²/s²")
    print(f"Media de ε: {np.mean(eps[-len(eps)//2:]):.4e} m²/s³")
    print(f"Media de νt: {np.mean(nu_t[-len(nu_t)//2:]):.4e} m²/s")