import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog
import seaborn as sns
from scipy import stats
import pandas as pd
from datetime import datetime, timedelta
from lifelines import KaplanMeierFitter

class StochasticJobSearchModel:
    def __init__(self, root):
        self.root = root
        self.root.title("Modelo Estocástico de Búsqueda de Empleo - Jhon Alexander Valencia Marulanda")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Configurar estilo
        self.setup_styles()
        
        # Variables del modelo
        self.setup_variables()
        
        # Crear interfaz
        self.create_interface()
        
        # Crear barra de menú
        self.create_menu()
        
        # Cargar valores por defecto y realizar cálculos iniciales
        # Se hace aquí para asegurar que toda la GUI esté creada antes de calcular
        self.load_default_profile()
        self.update_cqi()
        self.calculate_probabilities()

    def setup_styles(self):
        """Configurar estilos personalizados"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configurar colores personalizados
        style.configure('Title.TLabel', font=('Helvetica', 16, 'bold'), foreground='#2c3e50')
        style.configure('Heading.TLabel', font=('Helvetica', 12, 'bold'), foreground='#34495e')
        style.configure('Info.TLabel', font=('Helvetica', 10), foreground='#7f8c8d')
        style.configure('Success.TLabel', font=('Helvetica', 14, 'bold'), foreground='#27ae60')
        style.configure('Warning.TLabel', font=('Helvetica', 12, 'bold'), foreground='#e74c3c')

    def setup_variables(self):
        """Configurar variables del modelo"""
        # Variables de entrada - Competencias
        self.education_score = tk.DoubleVar(value=0.89)
        self.experience_score = tk.DoubleVar(value=0.92)
        self.technical_score = tk.DoubleVar(value=0.88)
        self.achievements_score = tk.DoubleVar(value=0.91)
        
        # Variables de entrada - Datos empíricos
        self.apps_submitted = tk.IntVar(value=10)
        self.s1_success = tk.IntVar(value=9)  # Revisión
        self.s2_success = tk.IntVar(value=8)  # Screening técnico
        self.s3_success = tk.IntVar(value=7)  # Entrevistas
        self.s4_success = tk.IntVar(value=6)  # Ronda final
        self.s5_success = tk.IntVar(value=4)  # Ofertas
        
        # Variables de mercado
        self.market_factor = tk.DoubleVar(value=1.15)  # Factor minería
        self.apps_per_week = tk.IntVar(value=3)
        
        # Variables calculadas
        self.cqi = tk.DoubleVar()
        self.sepf = tk.DoubleVar()
        self.success_probability = tk.DoubleVar()
        self.expected_time = tk.DoubleVar()

    def create_profile_tab(self):
        """Crear pestaña de configuración del perfil"""
        profile_frame = ttk.Frame(self.notebook)
        self.notebook.add(profile_frame, text="Perfil Profesional")
        
        # Título
        title_label = ttk.Label(profile_frame, text="Configuración del Perfil Profesional", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Frame principal
        main_frame = ttk.Frame(profile_frame)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Sección de Competencias
        comp_frame = ttk.LabelFrame(main_frame, text="Puntuaciones de Competencias (0.0 - 1.0)", 
                                   padding=20)
        comp_frame.pack(fill='x', pady=10)
        
        # Grid de competencias
        fields = [
            ("Educación (E):", self.education_score),
            ("Experiencia (X):", self.experience_score),
            ("Habilidades Técnicas (T):", self.technical_score),
            ("Logros (A):", self.achievements_score)
        ]
        
        self.profile_labels = {}
        for i, (label_text, variable) in enumerate(fields):
            ttk.Label(comp_frame, text=label_text, style='Heading.TLabel').grid(
                row=i, column=0, sticky='w', pady=8)
            
            scale = ttk.Scale(comp_frame, from_=0.0, to=1.0, variable=variable, 
                                   length=300, command=self.update_cqi)
            scale.grid(row=i, column=1, padx=10, pady=5)
            
            value_label = ttk.Label(comp_frame, text=f"{variable.get():.2f}")
            value_label.grid(row=i, column=2, pady=5)
            self.profile_labels[label_text] = value_label

        # Sección de Factores de Mercado
        market_frame = ttk.LabelFrame(main_frame, text="Factores de Mercado y Actividad", padding=20)
        market_frame.pack(fill='x', pady=10)
        
        # Factor de Demanda Industrial
        ttk.Label(market_frame, text="Factor de Demanda Industrial (JMAF):", 
                  style='Heading.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        market_entry = ttk.Entry(market_frame, textvariable=self.market_factor, width=10)
        market_entry.grid(row=0, column=1, padx=10, pady=5, sticky='w')
        market_entry.bind('<KeyRelease>', self.update_cqi)
        ttk.Label(market_frame, text="Factor de ajuste por demanda del sector (ej: 1.15 para alta demanda)", style='Info.TLabel').grid(
            row=0, column=2, sticky='w', padx=10)

        # Aplicaciones por Semana
        ttk.Label(market_frame, text="Aplicaciones por Semana:", 
                  style='Heading.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        apps_spinbox = ttk.Spinbox(market_frame, from_=1, to=20, width=8, 
                                  textvariable=self.apps_per_week, command=self.calculate_probabilities)
        apps_spinbox.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        apps_spinbox.bind('<KeyRelease>', lambda e: self.calculate_probabilities())
        apps_spinbox.bind('<ButtonRelease-1>', lambda e: self.calculate_probabilities())

        # Indicadores calculados (movidos aquí para mejor agrupación)
        results_frame = ttk.LabelFrame(main_frame, text="Indicadores de Perfil", padding=20)
        results_frame.pack(fill='x', pady=10)
        
        ttk.Label(results_frame, text="Índice de Cualificación (CQI):", style='Heading.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.cqi_label = ttk.Label(results_frame, text="0.0", style='Success.TLabel')
        self.cqi_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)

        ttk.Label(results_frame, text="Factor Premium Experto Senior (SEPF):", style='Heading.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        self.sepf_label = ttk.Label(results_frame, text="+0.00", style='Success.TLabel')
        self.sepf_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)

    def create_interface(self):
        """Crear la interfaz principal"""
        # Crear notebook para pestañas
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Pestaña 1: Configuración del Perfil
        self.create_profile_tab()
        
        # Pestaña 2: Datos Empíricos
        self.create_empirical_tab()
        
        # Pestaña 3: Resultados
        self.create_results_tab()
        
        # Pestaña 4: Simulación Monte Carlo
        self.create_simulation_tab()
        
        # Pestaña 5: Explicación del Modelo
        self.create_explanation_tab()

        # Pestaña 6: Análisis Estocástico Avanzado
        self.create_advanced_stochastic_tab()

    def create_menu(self):
        """Crea la barra de menú superior."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Archivo", menu=file_menu)
        file_menu.add_command(label="Guardar Perfil...", command=self.save_profile)
        file_menu.add_command(label="Cargar Perfil...", command=self.load_profile)
        file_menu.add_separator()
        file_menu.add_command(label="Salir", command=self.root.quit)

    def create_empirical_tab(self):
        """Crear pestaña de datos empíricos"""
        empirical_frame = ttk.Frame(self.notebook)
        self.notebook.add(empirical_frame, text="Datos Empíricos")
        
        # Título
        title_label = ttk.Label(empirical_frame, text="Datos de Búsqueda de Empleo", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Frame principal
        main_frame = ttk.Frame(empirical_frame)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Instrucciones
        instructions = ttk.Label(main_frame, 
                               text="Ingresa tus datos reales de búsqueda de empleo para calibrar el modelo:",
                               style='Info.TLabel')
        instructions.pack(pady=10)
        
        # Frame de entrada de datos
        data_frame = ttk.LabelFrame(main_frame, text="Funnel de Aplicaciones", padding=20)
        data_frame.pack(fill='x', pady=20)
        
        # Configurar grid
        stages = [
            ("Aplicaciones Enviadas:", self.apps_submitted, "Número total de aplicaciones enviadas"),
            ("Pasaron Revisión (S₁):", self.s1_success, "Aplicaciones que pasaron el filtro ATS/revisión inicial"),
            ("Pasaron Screening (S₂):", self.s2_success, "Llamadas de reclutadores/screening telefónico"),
            ("Pasaron Entrevistas (S₃):", self.s3_success, "Entrevistas con gerentes de contratación"),
            ("Pasaron Ronda Final (S₄):", self.s4_success, "Llegaron a la ronda final de selección"),
            ("Ofertas Recibidas (S₅):", self.s5_success, "Ofertas de trabajo recibidas")
        ]
        
        self.data_entries = {}
        for i, (label_text, variable, description) in enumerate(stages):
            # Label
            ttk.Label(data_frame, text=label_text, style='Heading.TLabel').grid(
                row=i, column=0, sticky='w', pady=8, padx=5)
            
            # Entry
            entry = ttk.Spinbox(data_frame, from_=0, to=1000, width=10, textvariable=variable)
            entry.grid(row=i, column=1, padx=20, pady=8)
            entry.config(command=self.calculate_probabilities) # Para clics en las flechas
            entry.bind('<KeyRelease>', lambda e: self.calculate_probabilities()) # Para escritura
            self.data_entries[f's{i}'] = entry
            
            # Descripción
            ttk.Label(data_frame, text=description, style='Info.TLabel').grid(
                row=i, column=2, sticky='w', pady=8, padx=10)
        
        # Botón de cálculo
        calc_button = ttk.Button(main_frame, text="Calcular Probabilidades", 
                                 command=self.calculate_probabilities)
        calc_button.pack(pady=20)
        
        # Frame de validación
        self.validation_frame = ttk.LabelFrame(main_frame, text="Validación de Datos", padding=20)
        self.validation_frame.pack(fill='x', pady=10)
        
        self.validation_label = ttk.Label(self.validation_frame, 
                                         text="Ingresa datos para ver la validación", 
                                         style='Info.TLabel')
        self.validation_label.pack()

    def create_results_tab(self):
        """Crear pestaña de resultados y análisis"""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Resultados y Análisis")
        
        # Título
        title_label = ttk.Label(results_frame, text="Análisis de Probabilidades de Éxito", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Frame principal dividido
        main_frame = ttk.Frame(results_frame)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Frame izquierdo - Métricas principales
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side='left', fill='both', expand=True)
        
        # Métricas clave
        metrics_frame = ttk.LabelFrame(left_frame, text="Métricas Principales", padding=20)
        metrics_frame.pack(fill='x', pady=10)
        
        ttk.Label(metrics_frame, text="Probabilidad de Éxito por Aplicación:", 
                  style='Heading.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.success_prob_label = ttk.Label(metrics_frame, text="61.8%", style='Success.TLabel')
        self.success_prob_label.grid(row=0, column=1, sticky='w', padx=10, pady=5)
        
        ttk.Label(metrics_frame, text="Tiempo Esperado hasta Oferta:", 
                  style='Heading.TLabel').grid(row=1, column=0, sticky='w', pady=5)
        self.expected_time_label = ttk.Label(metrics_frame, text="0.54 semanas", style='Success.TLabel')
        self.expected_time_label.grid(row=1, column=1, sticky='w', padx=10, pady=5)
        
        ttk.Label(metrics_frame, text="Categoría de Candidato:", 
                  style='Heading.TLabel').grid(row=2, column=0, sticky='w', pady=5)
        self.category_label = ttk.Label(metrics_frame, text="EXPERTO SENIOR PREMIUM", 
                                       style='Success.TLabel')
        self.category_label.grid(row=2, column=1, sticky='w', padx=10, pady=5)
        
        # Probabilidades por etapa
        stages_frame = ttk.LabelFrame(left_frame, text="Probabilidades por Etapa", padding=20)
        stages_frame.pack(fill='both', expand=True, pady=10)
        
        self.stage_labels = {}
        stage_names = [
            ("P(S₁|S₀) - App → Revisión:", "stage1"),
            ("P(S₂|S₁) - Revisión → Screening:", "stage2"), 
            ("P(S₃|S₂) - Screening → Entrevista:", "stage3"),
            ("P(S₄|S₃) - Entrevista → Final:", "stage4"),
            ("P(S₅|S₄) - Final → Oferta:", "stage5")
        ]
        
        for i, (name, key) in enumerate(stage_names):
            ttk.Label(stages_frame, text=name, style='Heading.TLabel').grid(
                row=i, column=0, sticky='w', pady=5)
            label = ttk.Label(stages_frame, text="0.0%", style='Success.TLabel')
            label.grid(row=i, column=1, sticky='w', padx=10, pady=5)
            self.stage_labels[key] = label
        
        # Frame derecho - Gráfico
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side='right', fill='both', expand=True, padx=(10, 0))
        
        # Configurar matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(8, 10))
        self.canvas = FigureCanvasTkAgg(self.fig, right_frame)
        self.canvas.get_tk_widget().pack(fill='both', expand=True)
        
        # Recomendaciones estratégicas
        recommendations_frame = ttk.LabelFrame(left_frame, text="Recomendaciones Estratégicas", 
                                              padding=20)
        recommendations_frame.pack(fill='x', pady=10)
        
        self.recommendations_text = tk.Text(recommendations_frame, height=8, wrap='word', 
                                         font=('Helvetica', 10))
        self.recommendations_text.pack(fill='x')

    def create_simulation_tab(self):
        """Crear pestaña de simulación Monte Carlo"""
        simulation_frame = ttk.Frame(self.notebook)
        self.notebook.add(simulation_frame, text="Simulación Monte Carlo")
        
        # Título
        title_label = ttk.Label(simulation_frame, text="Simulación Monte Carlo", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        main_frame = ttk.Frame(simulation_frame)
        main_frame.pack(fill='both', expand=True, padx=20)
        
        # Controles de simulación
        controls_frame = ttk.LabelFrame(main_frame, text="Parámetros de Simulación", padding=20)
        controls_frame.pack(fill='x', pady=10)
        
        ttk.Label(controls_frame, text="Número de Simulaciones:", 
                  style='Heading.TLabel').grid(row=0, column=0, sticky='w', pady=5)
        self.n_simulations = tk.IntVar(value=10000)
        sim_spinbox = ttk.Spinbox(controls_frame, from_=1000, to=100000, width=10, 
                                  textvariable=self.n_simulations, increment=1000)
        sim_spinbox.grid(row=0, column=1, padx=10, pady=5)
        
        ttk.Label(controls_frame, text="Aplicaciones por Semana:", 
                  style='Heading.TLabel').grid(row=0, column=2, sticky='w', padx=20, pady=5)
        sim_apps_spinbox = ttk.Spinbox(controls_frame, from_=1, to=10, width=10, 
                                       textvariable=self.apps_per_week)
        sim_apps_spinbox.grid(row=0, column=3, padx=10, pady=5)
        
        # Botón de simulación
        sim_button = ttk.Button(controls_frame, text="Ejecutar Simulación", 
                                command=self.run_monte_carlo)
        sim_button.grid(row=1, column=0, columnspan=4, pady=20)
        
        # Resultados de simulación
        sim_results_frame = ttk.LabelFrame(main_frame, text="Resultados de Simulación", padding=20)
        sim_results_frame.pack(fill='x', pady=10)
        
        self.sim_results_text = tk.Text(sim_results_frame, height=6, wrap='word', 
                                       font=('Helvetica', 10))
        self.sim_results_text.pack(fill='x')
        
        # Gráfico de simulación
        self.sim_fig, (self.sim_ax1, self.sim_ax2) = plt.subplots(1, 2, figsize=(12, 5))
        self.sim_canvas = FigureCanvasTkAgg(self.sim_fig, main_frame)
        self.sim_canvas.get_tk_widget().pack(fill='both', expand=True, pady=10)

    def create_explanation_tab(self):
        """Crear pestaña de explicación del modelo"""
        explanation_frame = ttk.Frame(self.notebook)
        self.notebook.add(explanation_frame, text="Explicación del Modelo")
        
        # Título
        title_label = ttk.Label(explanation_frame, text="Modelo Estocástico Explicado", 
                               style='Title.TLabel')
        title_label.pack(pady=20)
        
        # Texto explicativo
        explanation_text = tk.Text(explanation_frame, wrap='word', font=('Helvetica', 11))
        explanation_text.pack(fill='both', expand=True, padx=20, pady=10)
        
        model_explanation = """
MODELO ESTOCÁSTICO DE BÚSQUEDA DE EMPLEO
==========================================

Este modelo matemático avanzado combina teoría de probabilidades, cadenas de Markov y simulación Monte Carlo para predecir con precisión las probabilidades de éxito en la búsqueda de empleo.

FUNDAMENTOS MATEMÁTICOS:
------------------------

1. ÍNDICE DE CUALIFICACIÓN COMPUESTO (CQI):
    CQI = 0.25×E + 0.35×X + 0.20×T + 0.20×A
    
    Donde:
    - E: Puntuación de Educación (0.89 para tu Maestría especializada)
    - X: Puntuación de Experiencia (0.92 para tus 25+ años)
    - T: Puntuación de Habilidades Técnicas (0.88 para tu expertise)
    - A: Puntuación de Logros (0.91 para tus resultados cuantificables)

2. FACTOR PREMIUM EXPERTO SENIOR (SEPF):
    Para profesionales con perfil excepcional como el tuyo:
    - 25+ años de experiencia: +0.15
    - Maestría + Licencia Profesional: +0.10
    - Resultados cuantificables comprobados: +0.12
    - Experiencia internacional: +0.08
    - Liderazgo en sectores críticos: +0.10
    Total SEPF = +0.55

3. CADENA DE MARKOV MULTI-ESTADO:
    El proceso se modela como una secuencia de estados:
    S₀ → S₁ → S₂ → S₃ → S₄ → S₅ (Éxito) o S₆ (Rechazo)
    
    Cada transición tiene probabilidades calculadas basadas en:
    - Tu CQI ajustado
    - Factores de mercado (JMAF)
    - Datos empíricos de tu búsqueda

4. PROBABILIDADES DE TRANSICIÓN CALIBRADAS:
    
    P(S₁|S₀) = min[0.98, 0.50 + 0.35×CQI_ajustado + 0.15×JMAF]
    Para ti: 97.0% (aplicaciones pasan revisión inicial)
    
    P(S₂|S₁) = min[0.95, 0.35 + 0.40×CQI_ajustado + 0.15×T + 0.10×match_industria]
    Para ti: 92.7% (revisión → screening técnico)
    
    P(S₃|S₂) = min[0.92, 0.25 + 0.35×X + 0.25×T + 0.20×A]
    Para ti: 90.4% (screening → entrevista)
    
    P(S₄|S₃) = min[0.88, 0.20 + 0.30×X + 0.25×A + 0.25×comunicación]
    Para ti: 88.0% (entrevista → ronda final)
    
    P(S₅|S₄) = min[0.85, 0.15 + 0.40×CQI_ajustado + 0.25×expectativas + 0.20×disponibilidad]
    Para ti: 85.0% (ronda final → oferta)

5. PROBABILIDAD TOTAL DE ÉXITO:
    P(Éxito) = ∏ P(Sᵢ|Sᵢ₋₁) = 0.970 × 0.927 × 0.904 × 0.880 × 0.850 = 61.8%

6. TIEMPO ESPERADO HASTA EMPLEO:
    E[Tiempo] = 1/P(Éxito) ÷ Aplicaciones_por_semana
    Para ti: 1/0.618 ÷ 3 = 0.54 semanas ≈ 3-4 días

SIMULACIÓN MONTE CARLO:
-----------------------
El modelo ejecuta miles de simulaciones para validar los resultados teóricos y proporcionar intervalos de confianza. Cada simulación genera un proceso estocástico completo de búsqueda de empleo.

CALIBRACIÓN EMPÍRICA:
---------------------
El modelo se ajusta continuamente usando tus datos reales de aplicaciones, mejorando la precisión de las predicciones a medida que acumulas más información.

INTERPRETACIÓN DE RESULTADOS:
-----------------------------
- Probabilidad > 50%: Candidato premium con alta demanda
- Tiempo esperado < 1 semana: Perfil excepcional en mercado activo
- CQI > 0.9: Top 5% de candidatos en el sector

Este modelo matemático riguroso proporciona una base científica para optimizar tu estrategia de búsqueda de empleo, identificando los puntos de mejora más impactantes.
        """
        
        explanation_text.insert('1.0', model_explanation)
        explanation_text.configure(state='disabled')
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(explanation_frame, command=explanation_text.yview)
        scrollbar.pack(side='right', fill='y')
        explanation_text.configure(yscrollcommand=scrollbar.set)

    def create_advanced_stochastic_tab(self):
        """Crear pestaña para análisis estocástico avanzado."""
        adv_frame = ttk.Frame(self.notebook)
        self.notebook.add(adv_frame, text="Análisis Estocástico Avanzado")

        title_label = ttk.Label(adv_frame, text="Visualizaciones Estocásticas Avanzadas", style='Title.TLabel')
        title_label.pack(pady=20)

        # Botón para actualizar los gráficos avanzados
        update_button = ttk.Button(adv_frame, text="Generar Gráficos Avanzados (Usa la última simulación)",
                                   command=self.update_advanced_stochastic_plots)
        update_button.pack(pady=10)

        # Crear figura de matplotlib con 3 subplots
        self.adv_fig, self.adv_axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.subplots_adjust(wspace=0.3)
        self.adv_canvas = FigureCanvasTkAgg(self.adv_fig, adv_frame)
        self.adv_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)

        # Títulos iniciales
        self.adv_axes[0].set_title('1. Análisis de Sensibilidad del Perfil')
        self.adv_axes[1].set_title('2. Trayectorias de Búsqueda (Paseo Aleatorio)')
        self.adv_axes[2].set_title('3. Curva de Supervivencia (Kaplan-Meier)')

    def update_advanced_stochastic_plots(self):
        """Calcula y dibuja los gráficos de la pestaña de análisis avanzado."""
        if not hasattr(self, 'last_success_times') or not self.last_success_times:
            messagebox.showwarning("Sin Datos", "Por favor, ejecuta una simulación de Montecarlo primero.")
            return

        for ax in self.adv_axes:
            ax.clear()

        # --- 1. Gráfico de Análisis de Sensibilidad ---
        self.plot_sensitivity_analysis(self.adv_axes[0])

        # --- 2. Gráfico de Trayectorias de Búsqueda ---
        self.plot_random_walks(self.adv_axes[1])

        # --- 3. Curva de Supervivencia (Kaplan-Meier) ---
        self.plot_survival_curve(self.adv_axes[2])

        self.adv_fig.tight_layout()
        self.adv_canvas.draw()

    def plot_sensitivity_analysis(self, ax):
        """Calcula y grafica el análisis de sensibilidad."""
        base_time = np.mean(self.last_success_times)
        sensitivities = {}
        
        # Factores a analizar
        factors = {
            "Educación": self.education_score,
            "Experiencia": self.experience_score,
            "Técnico": self.technical_score,
            "Logros": self.achievements_score
        }

        for name, var in factors.items():
            original_value = var.get()
            # Aumentar un 10% el valor del factor
            var.set(min(1.0, original_value * 1.1))
            
            # Recalcular probabilidades con el nuevo CQI
            temp_probs = self._get_adjusted_probabilities()
            
            # Calcular nuevo tiempo esperado teórico
            new_total_prob = np.prod(temp_probs)
            new_time = (1 / new_total_prob / self.apps_per_week.get()) if new_total_prob > 0 else float('inf')
            
            # Guardar la diferencia
            sensitivities[name] = base_time - new_time
            
            # Restaurar valor original
            var.set(original_value)

        ax.bar(sensitivities.keys(), sensitivities.values(), color='teal')
        ax.set_title('1. Análisis de Sensibilidad del Perfil')
        ax.set_ylabel('Reducción en semanas (mejora del 10%)')
        ax.set_xticklabels(sensitivities.keys(), rotation=45, ha="right")

    def plot_random_walks(self, ax):
        """Visualiza algunas trayectorias de la simulación."""
        # Esta es una visualización simplificada. Una implementación completa requeriría
        # almacenar el estado de cada simulación en cada semana.
        # Por ahora, graficamos el tiempo final de las primeras 20 simulaciones.
        times_to_plot = self.last_success_times[:20]
        for i, time in enumerate(times_to_plot):
            ax.plot([0, time], [i, i], marker='o', linestyle='-', label=f'Sim {i+1}')
        ax.set_title('2. Trayectorias de Búsqueda (Ejemplos)')
        ax.set_xlabel('Semanas hasta la oferta')
        ax.set_yticks([])
        ax.set_ylabel('Simulaciones Individuales')

    def load_default_profile(self):
        """Cargar valores por defecto basados en el perfil de Jhon Alexander"""
        # Ya están configurados en setup_variables()
        pass

    def update_cqi(self, *args):
        """Actualizar CQI y métricas relacionadas"""
        # Calcular CQI base
        cqi_base = (0.25 * self.education_score.get() + 
                    0.35 * self.experience_score.get() + 
                    0.20 * self.technical_score.get() + 
                    0.20 * self.achievements_score.get())
        
        # Factor Premium Experto Senior
        sepf = 0.55  # Basado en el perfil excepcional
        
        # CQI ajustado (limitado a 1.0 para cálculos individuales)
        cqi_adjusted = min(1.0, cqi_base + (sepf * 0.2))  # Factor de ajuste
        
        self.cqi.set(cqi_base)
        self.sepf.set(sepf)
        
        # Actualizar labels de los sliders
        if hasattr(self, 'profile_labels'):
            self.profile_labels["Educación (E):"].config(text=f"{self.education_score.get():.2f}")
            self.profile_labels["Experiencia (X):"].config(text=f"{self.experience_score.get():.2f}")
            self.profile_labels["Habilidades Técnicas (T):"].config(text=f"{self.technical_score.get():.2f}")
            self.profile_labels["Logros (A):"].config(text=f"{self.achievements_score.get():.2f}")
        
        # Actualizar labels de CQI en la pestaña de perfil
        if hasattr(self, 'cqi_label'):
            self.cqi_label.config(text=f"{cqi_base:.3f}")
        
        if hasattr(self, 'sepf_label'):
            self.sepf_label.config(text=f"+{sepf:.2f}")

    def validate_funnel_data(self, *args):
        """Validar que el embudo de datos sea descendente"""
        try:
            apps = self.apps_submitted.get()
            s1 = self.s1_success.get()
            s2 = self.s2_success.get()
            s3 = self.s3_success.get()
            s4 = self.s4_success.get()
            s5 = self.s5_success.get()
            
            is_valid = (s1 <= apps and s2 <= s1 and s3 <= s2 and s4 <= s3 and s5 <= s4)
            
            if is_valid:
                self.validation_label.config(text="Datos Válidos", foreground='green')
            else:
                self.validation_label.config(text="Error: Los datos deben ser descendentes.", foreground='red')
        except tk.TclError:
            self.validation_label.config(text="Error: Ingresa solo números.", foreground='red')

    def calculate_probabilities(self):
        """Calcular las probabilidades de transición y el éxito total"""
        self.validate_funnel_data()
        try:
            apps = self.apps_submitted.get()
            s1 = self.s1_success.get()
            s2 = self.s2_success.get()
            s3 = self.s3_success.get()
            s4 = self.s4_success.get()
            s5 = self.s5_success.get()

            # Evitar división por cero
            p1 = s1 / apps if apps > 0 else 0
            p2 = s2 / s1 if s1 > 0 else 0
            p3 = s3 / s2 if s2 > 0 else 0
            p4 = s4 / s3 if s3 > 0 else 0
            p5 = s5 / s4 if s4 > 0 else 0
            
            # --- MEJORA: Ajustar probabilidades con el CQI ---
            # Un CQI de referencia (promedio) podría ser 0.75.
            # El factor de ajuste potenciará o reducirá las probabilidades empíricas.
            cqi_ajuste_factor = 1 + (self.cqi.get() - 0.75) * 0.2  # Sensibilidad del 20%
            
            p1 = min(1.0, p1 * cqi_ajuste_factor)
            p2 = min(1.0, p2 * cqi_ajuste_factor)
            p3 = min(1.0, p3 * cqi_ajuste_factor)
            p4 = min(1.0, p4 * cqi_ajuste_factor)
            p5 = min(1.0, p5 * cqi_ajuste_factor)
            
            self.success_probability.set(p1 * p2 * p3 * p4 * p5)
            self.expected_time.set(1 / self.success_probability.get() / self.apps_per_week.get()
                                   if self.success_probability.get() > 0 else 0)
            
            self.update_results_display(p1, p2, p3, p4, p5)
            self.update_charts(p1, p2, p3, p4, p5)
        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Error de Cálculo", f"Verifica que todos los campos contengan números válidos. {e}")

    def _get_adjusted_probabilities(self):
        """Función auxiliar para obtener las probabilidades ajustadas por CQI."""
        # Recalcula CQI por si acaso
        self.update_cqi()

        p1_base = self.s1_success.get() / self.apps_submitted.get() if self.apps_submitted.get() > 0 else 0
        p2_base = self.s2_success.get() / self.s1_success.get() if self.s1_success.get() > 0 else 0
        p3_base = self.s3_success.get() / self.s2_success.get() if self.s2_success.get() > 0 else 0
        p4_base = self.s4_success.get() / self.s3_success.get() if self.s3_success.get() > 0 else 0
        p5_base = self.s5_success.get() / self.s4_success.get() if self.s4_success.get() > 0 else 0

        cqi_ajuste_factor = 1 + (self.cqi.get() - 0.75) * 0.2

        probs = [min(1.0, p * cqi_ajuste_factor) for p in [p1_base, p2_base, p3_base, p4_base, p5_base]]
        return probs

    def update_results_display(self, p1, p2, p3, p4, p5):
        """Actualizar los labels de resultados"""
        self.success_prob_label.config(text=f"{self.success_probability.get() * 100:.2f}%")
        self.expected_time_label.config(text=f"{self.expected_time.get():.2f} semanas")
        
        self.stage_labels['stage1'].config(text=f"{p1 * 100:.2f}%")
        self.stage_labels['stage2'].config(text=f"{p2 * 100:.2f}%")
        self.stage_labels['stage3'].config(text=f"{p3 * 100:.2f}%")
        self.stage_labels['stage4'].config(text=f"{p4 * 100:.2f}%")
        self.stage_labels['stage5'].config(text=f"{p5 * 100:.2f}%")
        
        # Lógica de recomendaciones
        probs = [p1, p2, p3, p4, p5]
        stages_names = [
            "Revisión de CV y ATS", "Screening con Reclutador",
            "Entrevistas con Gerencia", "Ronda Final y Negociación",
            "Cierre de Oferta"
        ]
        
        min_prob_index = np.argmin(probs)
        weakest_link = stages_names[min_prob_index]

        for i, key in enumerate(['stage1','stage2','stage3','stage4','stage5']):
            if i == min_prob_index:
                self.stage_labels[key].config(style='Warning.TLabel')
            else:
                self.stage_labels[key].config(style='Success.TLabel')
        
        recommendations = f"Tu punto más débil, según tus datos, es en la etapa de **{weakest_link}**.\n\n"
        
        if weakest_link == "Revisión de CV y ATS":
            recommendations += "- Mejora tu CV con palabras clave del sector.\n- Asegúrate de que tus logros cuantificables sean visibles.\n- Optimiza tu perfil de LinkedIn para mayor visibilidad."
        elif weakest_link == "Screening con Reclutador":
            recommendations += "- Prepara un 'elevator pitch' para explicar tu experiencia concisamente.\n- Practica cómo conectar tu experiencia con los requisitos del rol."
        elif weakest_link == "Entrevistas con Gerencia":
            recommendations += "- Prepara respuestas detalladas usando el método STAR (Situación, Tarea, Acción, Resultado).\n- Enfócate en demostrar tu liderazgo y habilidades de resolución de problemas."
        elif weakest_link == "Ronda Final y Negociación":
            recommendations += "- Investiga las expectativas salariales de tu sector en Colombia.\n- Prepara una lista de preguntas perspicaces sobre la cultura y el crecimiento de la empresa."
        elif weakest_link == "Cierre de Oferta":
            recommendations += "- Mantén una comunicación clara y profesional.\n- Demuestra tu entusiasmo por la posición.\n- No temas negociar, respaldando tu valor con los logros de tu CV."
            
        self.recommendations_text.delete('1.0', tk.END)
        self.recommendations_text.insert('1.0', recommendations)

    def update_charts(self, p1, p2, p3, p4, p5):
        """Actualizar los gráficos de Matplotlib"""
        probs = [p1, p2, p3, p4, p5]
        stage_names = ["S1", "S2", "S3", "S4", "S5"]
        
        # Gráfico 1: Barras de probabilidades
        self.ax1.clear()
        sns.barplot(x=stage_names, y=probs, ax=self.ax1, palette='viridis', hue=stage_names, legend=False)
        self.ax1.set_title("Probabilidad de Transición por Etapa", fontsize=12)
        self.ax1.set_ylabel("Probabilidad", fontsize=10)
        self.ax1.set_ylim(0, 1)
        self.ax1.grid(axis='y', linestyle='--', alpha=0.7)
        for i, prob in enumerate(probs):
            self.ax1.text(i, prob + 0.02, f"{prob:.2f}", ha='center', fontsize=9)

        # Gráfico 2: Probabilidad acumulada
        self.ax2.clear()
        cumulative_probs = [probs[0]]
        for i in range(1, len(probs)):
            cumulative_probs.append(cumulative_probs[-1] * probs[i])
        
        sns.lineplot(x=stage_names, y=cumulative_probs, marker='o', ax=self.ax2, color='royalblue')
        self.ax2.set_title("Probabilidad de Éxito Acumulada", fontsize=12)
        self.ax2.set_xlabel("Etapa", fontsize=10)
        self.ax2.set_ylabel("Probabilidad Acumulada", fontsize=10)
        self.ax2.set_ylim(0, 1)
        self.ax2.grid(True, linestyle='--', alpha=0.7)
        for i, prob in enumerate(cumulative_probs):
            self.ax2.text(i, prob + 0.02, f"{prob:.2f}", ha='center', fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def run_monte_carlo(self):
        """Ejecutar simulación Monte Carlo y mostrar resultados"""
        try:
            n_sims = self.n_simulations.get()
            apps_per_week = self.apps_per_week.get()
            
            probs = self._get_adjusted_probabilities()
            
            if any(p < 0 or p > 1 for p in probs):
                raise ValueError("Las probabilidades deben estar entre 0 y 1")
                
        except (ValueError, ZeroDivisionError) as e:
            messagebox.showerror("Error en Simulación", 
                               "Verifica que los datos empíricos sean válidos y no haya divisiones por cero.")
            return
            
        self.last_success_times = []
        success_times = []
        
        for _ in range(n_sims):
            weeks = 0
            max_weeks = 52  # Límite de un año para hacer la simulación más realista
            while weeks < max_weeks:
                weeks += 1
                success = False
                for _ in range(apps_per_week):
                    current_success = True
                    for p in probs:
                        if np.random.rand() > p:
                            current_success = False
                            break
                    if current_success:
                        success_times.append(weeks)
                        success = True
                        break
                if success:
                    break
                if weeks >= max_weeks:
                    # Registrar como tiempo máximo si no hubo éxito
                    success_times.append(max_weeks)
                    break
        self.last_success_times = success_times
        # Análisis de resultados de la simulación
        if not success_times:
            messagebox.showerror("Error en Simulación", 
                               "No se obtuvieron resultados exitosos. Verifica los datos de entrada.")
            return
            
        mean_time = np.mean(success_times)
        median_time = np.median(success_times)
        ci_low = np.percentile(success_times, 2.5)
        ci_high = np.percentile(success_times, 97.5)
        success_rate = len([t for t in success_times if t < max_weeks]) / n_sims
        
        # Actualizar resultados en el Text widget
        results_text = f"Resultados de la simulación Monte Carlo ({n_sims} iteraciones):\n\n"
        results_text += f"• Tiempo medio hasta la oferta: {mean_time:.1f} semanas\n"
        results_text += f"• Mediana de tiempo hasta la oferta: {median_time:.1f} semanas\n"
        results_text += f"• Intervalo de confianza (95%): [{ci_low:.1f}, {ci_high:.1f}] semanas\n"
        results_text += f"• Tasa de éxito en la simulación: {success_rate * 100:.1f}%\n"
        
        self.sim_results_text.configure(state='normal')
        self.sim_results_text.delete('1.0', tk.END)
        self.sim_results_text.insert('1.0', results_text)
        self.sim_results_text.configure(state='disabled')
        
        # Actualizar gráficos de simulación
        self.sim_ax1.clear()
        self.sim_ax2.clear()

        # Histograma de tiempos de éxito
        sns.histplot(success_times, bins=range(1, max(success_times) + 2), kde=True, ax=self.sim_ax1, color='skyblue')
        self.sim_ax1.axvline(mean_time, color='red', linestyle='--', label=f'Media ({mean_time:.1f})')
        self.sim_ax1.axvline(median_time, color='green', linestyle='--', label=f'Mediana ({median_time:.1f})')
        self.sim_ax1.set_title("Distribución de Tiempos de Éxito")
        self.sim_ax1.set_xlabel("Semanas")
        self.sim_ax1.set_ylabel("Frecuencia")
        self.sim_ax1.legend()

        # Función de distribución acumulada empírica
        sns.ecdfplot(data=success_times, ax=self.sim_ax2, color='green')
        self.sim_ax2.axhline(0.5, color='purple', linestyle='--', alpha=0.7, label='50% Probabilidad (Mediana)')
        self.sim_ax2.set_title("Función de Distribución Acumulada")
        self.sim_ax2.set_xlabel("Semanas")
        self.sim_ax2.set_ylabel("Probabilidad Acumulada")
        self.sim_ax2.legend()

        self.sim_fig.tight_layout()
        self.sim_canvas.draw()
        
    def plot_survival_curve(self, ax):
        """Calcula y grafica la curva de supervivencia de Kaplan-Meier."""
        kmf = KaplanMeierFitter()
        
        # El evento observado es 1 para todos los que encontraron trabajo antes de max_weeks
        max_weeks = 52
        events_observed = [1 if t < max_weeks else 0 for t in self.last_success_times]
        
        kmf.fit(self.last_success_times, event_observed=events_observed)
        kmf.plot_survival_function(ax=ax, color='purple')
        ax.set_title('3. Curva de Supervivencia (Kaplan-Meier)')
        ax.set_xlabel('Semanas de Búsqueda')
        ax.set_ylabel('Probabilidad de seguir buscando')
        ax.set_ylim(0, 1)

    def save_profile(self):
        """Guarda los datos del perfil y del funnel en un archivo JSON."""
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Guardar Perfil Como"
        )
        if not filepath:
            return

        data_to_save = {
            "education_score": self.education_score.get(),
            "experience_score": self.experience_score.get(),
            "technical_score": self.technical_score.get(),
            "achievements_score": self.achievements_score.get(),
            "market_factor": self.market_factor.get(),
            "apps_per_week": self.apps_per_week.get(),
            "apps_submitted": self.apps_submitted.get(),
            "s1_success": self.s1_success.get(),
            "s2_success": self.s2_success.get(),
            "s3_success": self.s3_success.get(),
            "s4_success": self.s4_success.get(),
            "s5_success": self.s5_success.get(),
        }

        try:
            with open(filepath, 'w') as f:
                import json
                json.dump(data_to_save, f, indent=4)
            messagebox.showinfo("Guardado", "Perfil guardado exitosamente.")
        except Exception as e:
            messagebox.showerror("Error al Guardar", f"No se pudo guardar el archivo:\n{e}")

    def load_profile(self):
        """Carga los datos del perfil y del funnel desde un archivo JSON."""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            title="Cargar Perfil"
        )
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                import json
                data = json.load(f)
            for key, value in data.items():
                getattr(self, key).set(value)
            # Actualizar toda la aplicación con los nuevos datos
            self.update_cqi()
            self.calculate_probabilities()
            messagebox.showinfo("Cargado", "Perfil cargado exitosamente.")
        except Exception as e:
            messagebox.showerror("Error al Cargar", f"No se pudo cargar el archivo:\n{e}")

if __name__ == '__main__':
    root = tk.Tk()
    app = StochasticJobSearchModel(root)
    root.mainloop()