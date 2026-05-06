import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
import plotly.express as px

st.set_page_config(
    page_title="Finanzas Cuantitativas 2 | 1FIN04",
    page_icon="📊",
    layout="wide"
)

# ── estilos globales ────────────────────────────────────────────────────────
st.markdown("""
<style>
    .def-box {
        background: #0e1117;
        border-left: 4px solid #e63946;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
    }
    .teo-box {
        background: #0e1117;
        border-left: 4px solid #2196F3;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
    }
    .prop-box {
        background: #0e1117;
        border-left: 4px solid #4CAF50;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
    }
    .obs-box {
        background: #1a1a2e;
        border-left: 4px solid #FF9800;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
    }
    .ejemplo-box {
        background: #0d1f0d;
        border-left: 4px solid #66BB6A;
        padding: 14px 18px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0 16px 0;
    }
    .label-def   { color:#e63946; font-weight:700; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; }
    .label-teo   { color:#2196F3; font-weight:700; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; }
    .label-prop  { color:#4CAF50; font-weight:700; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; }
    .label-obs   { color:#FF9800; font-weight:700; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; }
    .label-ej    { color:#66BB6A; font-weight:700; font-size:0.8rem; letter-spacing:1px; text-transform:uppercase; }
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #e0e0e0;
        border-bottom: 1px solid #333;
        padding-bottom: 6px;
        margin: 24px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑAS PRINCIPALES
# ═══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📐 Independencia de V.A.",
    "🔗 Independencia de Vectores",
    "📊 Covarianza",
    "📈 Correlación",
    "🔔 Normal Multivariada",
    "✏️ Ejercicios resueltos"
])

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA 1 — INDEPENDENCIA DE VARIABLES ALEATORIAS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## Independencia de Variables Aleatorias")
    st.markdown(
        "Estudiamos cuándo el comportamiento de una variable aleatoria "
        "**no aporta información** sobre el comportamiento de otra."
    )

    teoria_tab, visual_tab = st.tabs(["📖 Teoría", "🎮 Visualización interactiva"])

    # ── TEORÍA ──────────────────────────────────────────────────────────────
    with teoria_tab:

        # 0) MOTIVACIÓN
        st.markdown('<div class="section-title">0 · Motivación</div>', unsafe_allow_html=True)
        st.markdown(
            "Sea $(X_1, X_2)$ un vector aleatorio. En las aplicaciones frecuentemente "
            "necesitamos variables que son **funciones** de ese vector."
        )
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo financiero</span>', unsafe_allow_html=True)
        st.latex(r"g(X_1, X_2) = X_1 \cdot X_2")
        st.markdown(
            "donde $X_1$ es el precio de una acción en USD y $X_2$ es el tipo de cambio "
            "USD/PEN. Entonces $g(X_1,X_2)$ es el precio en soles. "
            "En general: $g: \\mathbb{R}^2 \\to \\mathbb{R}$, $Y = g(X_1, X_2)$ es una variable aleatoria."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="def-box"><span class="label-def">Definición</span>', unsafe_allow_html=True)
        st.markdown("Sea $(X_1, X_2)$ un vector aleatorio. Se define:")
        st.latex(r"E\bigl[g(X_1,X_2)\bigr] = \iint_{\mathbb{R}^2} g(x_1,x_2)\,f_{1,2}(x_1,x_2)\,dx_1\,dx_2 \quad \text{(caso continuo)}")
        st.latex(r"E\bigl[g(X_1,X_2)\bigr] = \sum_{x_2}\sum_{x_1} g(x_1,x_2)\,p_{1,2}(x_1,x_2) \quad \text{(caso discreto)}")
        st.latex(r"\text{Var}\bigl(g(X_1,X_2)\bigr) = E\Bigl[\bigl(g(X_1,X_2) - E[g(X_1,X_2)]\bigr)^2\Bigr]")
        st.markdown("</div>", unsafe_allow_html=True)

        # 1) INDEPENDENCIA DE EVENTOS
        st.markdown('<div class="section-title">1 · Independencia de eventos</div>', unsafe_allow_html=True)
        st.markdown(
            "Antes de definir independencia de variables aleatorias, recordamos la noción "
            "para **eventos**, pues es el fundamento conceptual."
        )
        st.markdown('<div class="def-box"><span class="label-def">Definición</span>', unsafe_allow_html=True)
        st.markdown("Los eventos $A_1$ y $A_2$ son **independientes** si:")
        st.latex(r"\mathbb{P}(A_1 \mid A_2) = \mathbb{P}(A_1), \quad \mathbb{P}(A_2) > 0")
        st.latex(r"\mathbb{P}(A_2 \mid A_1) = \mathbb{P}(A_2), \quad \mathbb{P}(A_1) > 0")
        st.markdown("Equivalentemente (definición operacional):")
        st.latex(r"\mathbb{P}(A_1 \cap A_2) = \mathbb{P}(A_1)\cdot\mathbb{P}(A_2)")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "> 💡 **En palabras:** conocer que $A_2$ ocurrió no cambia la probabilidad de que $A_1$ ocurra, "
            "y viceversa. La segunda definición (producto) es la que se usa en la práctica porque "
            "no requiere condicionar."
        )

        # 2) INDEPENDENCIA DE VARIABLES ALEATORIAS
        st.markdown('<div class="section-title">2 · Independencia de variables aleatorias</div>', unsafe_allow_html=True)
        st.markdown(
            "Las variables aleatorias generan eventos naturalmente: "
            "dado un intervalo $I \\subset \\mathbb{R}$, el evento generado por $X_j$ es "
            "$A_{X_j} = \\{X_j \\in I\\}$."
        )
        st.markdown('<div class="def-box"><span class="label-def">Definición</span>', unsafe_allow_html=True)
        st.markdown(
            "$X_1$ y $X_2$ son **independientes** si **todos** los eventos generados "
            "por $X_1$ y $X_2$ son independientes; es decir, si:"
        )
        st.latex(r"\mathbb{P}(X_1 \in I_1 \mid X_2 \in I_2) = \mathbb{P}(X_1 \in I_1), \quad \forall\, I_1, I_2 \subset \mathbb{R}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "> 💡 **En palabras:** la información generada por la ocurrencia de **ningún** evento de $X_2$ "
            "modifica la probabilidad de ningún evento de $X_1$, y viceversa."
        )

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema — Criterio práctico (caso continuo)</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $(X_1, X_2)$ continuo con densidad conjunta $f_{1,2}$ y marginales $f_1$, $f_2$. "
            "$X_1$ y $X_2$ son independientes **si y solo si**:"
        )
        st.latex(r"f_{1,2}(x_1, x_2) = f_1(x_1)\cdot f_2(x_2), \quad \forall\, x_1 \in S_1,\; \forall\, x_2 \in S_2")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema — Criterio práctico (caso discreto)</span>', unsafe_allow_html=True)
        st.markdown("$X_1$ y $X_2$ son independientes **si y solo si**:")
        st.latex(r"p_{1,2}(x_1, x_2) = p_1(x_1)\cdot p_2(x_2), \quad \forall\, x_1 \in S_1,\; \forall\, x_2 \in S_2")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Consecuencia inmediata:** si $X_1$ y $X_2$ son independientes, "
            "la densidad condicional colapsa a la marginal:"
        )
        st.latex(r"f_{2|1}(x_2 \mid x_1) = \frac{f_{1,2}(x_1,x_2)}{f_1(x_1)} = \frac{f_1(x_1)\cdot f_2(x_2)}{f_1(x_1)} = f_2(x_2)")
        st.markdown("Saber el valor de $X_1$ no cambia la distribución de $X_2$.")

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo — Dos dados</span>', unsafe_allow_html=True)
        st.markdown("Sea $\\Omega = \\{1,2,3,4,5,6\\}^2$. $X_1$: resultado del dado 1, $X_2$: resultado del dado 2.")
        st.latex(r"p_{1,2}(x_1,x_2) = \frac{1}{36}, \quad p_1(x_1) = \frac{1}{6}, \quad p_2(x_2) = \frac{1}{6}")
        st.latex(r"p_{1,2}(x_1,x_2) = \frac{1}{36} = \frac{1}{6}\cdot\frac{1}{6} = p_1(x_1)\cdot p_2(x_2) \;\checkmark")
        st.markdown("Por el teorema, $X_1$ y $X_2$ son independientes.")
        st.markdown("</div>", unsafe_allow_html=True)

        # 3) DEPENDENCIA
        st.markdown('<div class="section-title">3 · Dependencia de variables aleatorias</div>', unsafe_allow_html=True)
        st.markdown('<div class="def-box"><span class="label-def">Definición</span>', unsafe_allow_html=True)
        st.markdown("$X_1$ y $X_2$ son **dependientes** si no son independientes.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="def-box"><span class="label-def">Definición — Determinación</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1$ y $X_2$ dependientes. Se dice que $X_1$ **determina completamente** "
            "a $X_2$ si existe $g: S_1 \\to \\mathbb{R}$ tal que $X_2 = g(X_1)$. "
            "En caso contrario, $X_1$ **determina parcialmente** a $X_2$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo financiero — Bono</span>', unsafe_allow_html=True)
            st.markdown("$X_1$: precio del bono. $X_2$: rendimiento del bono.")
            st.latex(r"X_1 = \frac{F}{1+X_2} = g(X_2)")
            st.markdown(
                "$X_2$ determina **completamente** a $X_1$. "
                "Y también $X_1 = h(X_2)^{-1}$, por lo que $X_1$ determina completamente a $X_2$. "
                "La relación es **biunívoca**."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo financiero — Call</span>', unsafe_allow_html=True)
            st.markdown("$X_1$: precio del subyacente. $X_2$: profit del call.")
            st.latex(r"X_2 = \max\{X_1 - K,\, 0\} - c = g(X_1)")
            st.markdown(
                "$X_1$ determina **completamente** a $X_2$. "
                "Pero si $X_2 = -c$, entonces $X_1 \\in (0, K]$ — no podemos recuperar $X_1$ exactamente. "
                "Luego $X_2$ determina **parcialmente** a $X_1$."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # 4) PROPIEDADES
        st.markdown('<div class="section-title">4 · Propiedades de variables independientes</div>', unsafe_allow_html=True)
        st.markdown('<div class="prop-box"><span class="label-prop">Propiedades</span>', unsafe_allow_html=True)
        st.markdown("Sean $X_1$ y $X_2$ **independientes**. Entonces:")
        st.latex(r"\text{(1)}\quad E[X_1 \mid X_2 = x_2] = E[X_1], \quad \forall\, x_2 \in S_2")
        st.latex(r"\text{(2)}\quad \text{Var}[X_1 \mid X_2 = x_2] = \text{Var}(X_1), \quad \forall\, x_2 \in S_2")
        st.latex(r"\text{(3)}\quad E(X_1 \cdot X_2) = E(X_1)\cdot E(X_2)")
        st.latex(r"\text{(4)}\quad E\bigl[g(X_1)\cdot h(X_2)\bigr] = E\bigl[g(X_1)\bigr]\cdot E\bigl[h(X_2)\bigr]")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "> 💡 **(1) y (2):** condicionar en $X_2$ no cambia ni la media ni la varianza de $X_1$ — "
            "la información de $X_2$ es irrelevante. **(3)** es consecuencia directa del criterio de "
            "factorización. **(4)** generaliza (3): la independencia se preserva bajo funciones."
        )

        # ADVERTENCIA CRÍTICA
        st.markdown('<div class="obs-box"><span class="label-obs">Advertencia importante</span>', unsafe_allow_html=True)
        st.markdown(
            "**Independencia $\\Rightarrow$ $\\text{Cov}(X_1,X_2)=0$**, pero la recíproca es **falsa** en general. "
            "Covarianza cero solo captura ausencia de relación **lineal**; puede existir dependencia no lineal "
            "perfecta con covarianza nula. "
            "**Excepción:** bajo normalidad conjunta, $\\text{Cov}=0 \\Leftrightarrow$ independencia. "
            "Esta distinción se verá en detalle en la pestaña de Correlación."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── VISUALIZACIÓN ───────────────────────────────────────────────────────

    # ── VISUALIZACIÓN PESTAÑA 1 ─────────────────────────────────────────────
    with visual_tab:
        st.markdown("### Herramientas visuales")
        viz1, viz2, viz3 = st.tabs([
            "🔍 ¿La densidad conjunta factoriza?",
            "📦 Condicional vs. marginal",
            "🎲 Verificando independencia con muestras"
        ])

        # VIZ 1 — FACTORIZACIÓN
        with viz1:
            st.markdown(
                "**El criterio práctico de independencia** dice que $X_1 \\perp X_2$ si y solo si "
                "$f_{1,2}(x_1,x_2) = f_1(x_1)\\cdot f_2(x_2)$.  \n"
                "Cuando esto ocurre, la densidad conjunta se puede 'separar' en dos factores "
                "que dependen cada uno de una sola variable. "
                "Visualmente, la superficie 3D de la densidad conjunta adopta una forma "
                "**simétrica sin inclinación** cuando las variables son independientes, "
                "y se **tuerce** cuando son dependientes.  \n\n"
                "Usa el control para pasar de independencia a dependencia y observa cómo cambia la forma."
            )
            col_ctrl, col_plot = st.columns([1, 2])
            with col_ctrl:
                dep_v1 = st.select_slider(
                    "Grado de dependencia entre X₁ y X₂",
                    options=["Independientes", "Dependencia débil", "Dependencia fuerte"],
                    value="Independientes", key="v1_dep"
                )
                rho_map_v1 = {"Independientes": 0.0, "Dependencia débil": 0.5, "Dependencia fuerte": 0.9}
                rho_v1 = rho_map_v1[dep_v1]

                st.markdown("---")
                if dep_v1 == "Independientes":
                    st.success(
                        "✅ **Independientes.**  \n"
                        "La densidad conjunta **factoriza**: $f_{1,2} = f_1 \\cdot f_2$.  \n"
                        "Conocer $X_1$ no aporta información sobre $X_2$."
                    )
                else:
                    st.error(
                        f"❌ **Dependientes.**  \n"
                        "La densidad conjunta **no factoriza**.  \n"
                        "Conocer $X_1$ cambia lo que sabemos sobre $X_2$."
                    )

            with col_plot:
                x = np.linspace(-3.5, 3.5, 70)
                y = np.linspace(-3.5, 3.5, 70)
                X_g, Y_g = np.meshgrid(x, y)
                pos = np.dstack((X_g, Y_g))
                cov_mat = np.array([[1, rho_v1], [rho_v1, 1]])
                rv = stats.multivariate_normal(mean=[0, 0], cov=cov_mat)
                Z = rv.pdf(pos)
                f1 = stats.norm(0, 1).pdf(x)
                f2 = stats.norm(0, 1).pdf(y)

                fig_v1 = go.Figure()
                fig_v1.add_trace(go.Surface(
                    x=x, y=y, z=Z,
                    colorscale="Viridis", opacity=0.88, showscale=False,
                    name="f₁,₂(x₁,x₂)"
                ))
                fig_v1.add_trace(go.Scatter3d(
                    x=x, y=np.full_like(x, 3.7), z=f1 * 0.45,
                    mode='lines', line=dict(color='#e63946', width=5),
                    name="f₁(x₁) — marginal"
                ))
                fig_v1.add_trace(go.Scatter3d(
                    x=np.full_like(y, -3.7), y=y, z=f2 * 0.45,
                    mode='lines', line=dict(color='#FF9800', width=5),
                    name="f₂(x₂) — marginal"
                ))
                fig_v1.update_layout(
                    scene=dict(
                        xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x₁,x₂)",
                        bgcolor='rgba(0,0,0,0)',
                        camera=dict(eye=dict(x=1.6, y=-1.6, z=1.2))
                    ),
                    height=460,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(font=dict(color='white'))
                )
                st.plotly_chart(fig_v1, use_container_width=True)
                st.caption(
                    "Rojo: densidad marginal $f_1(x_1)$. Naranja: densidad marginal $f_2(x_2)$. "
                    "Cuando son independientes, la superficie verde es exactamente el producto "
                    "de esas dos curvas en cada punto. Al haber dependencia, la superficie se tuerce "
                    "y ese producto ya no reconstruye la conjunta."
                )

        # VIZ 2 — CONDICIONAL VS MARGINAL
        with viz2:
            st.markdown(
                "**Propiedad clave de la independencia:**  \n"
                "Si $X_1 \\perp X_2$, entonces $f_{2|1}(x_2 \\mid x_1) = f_2(x_2)$ para todo $x_1$.  \n\n"
                "En palabras: la distribución de $X_2$ **no cambia** cuando te dicen el valor de $X_1$. "
                "Saber el valor de $X_1$ es completamente irrelevante para $X_2$.  \n\n"
                "En la visualización compararás la densidad marginal de $X_2$ (lo que sabes sin "
                "información adicional) con la densidad condicional dado $X_1 = x_1$ "
                "(lo que sabes una vez que conoces $X_1$). "
                "Si son independientes, ambas curvas deben ser **idénticas**."
            )
            col_c1, col_c2 = st.columns([1, 2])
            with col_c1:
                indep_v2 = st.radio(
                    "¿Son X₁ y X₂ independientes?",
                    ["Sí — independientes", "No — dependientes"],
                    key="v2_indep"
                )
                rho_v2 = 0.0 if indep_v2 == "Sí — independientes" else 0.8
                x1_cond = st.slider(
                    "Valor observado de X₁",
                    -3.0, 3.0, 1.5, 0.1, key="v2_x1"
                )
                st.markdown("---")
                mu_cond_v2 = rho_v2 * x1_cond
                sigma_cond_v2 = np.sqrt(max(1 - rho_v2**2, 1e-9))
                st.markdown("**Distribución marginal** $X_2 \\sim N(0,1)$:")
                st.markdown(f"Media = `0.000` | σ = `1.000`")
                st.markdown(f"**Distribución condicional** $X_2 \\mid X_1={x1_cond:.1f}$:")
                st.markdown(f"Media = `{mu_cond_v2:.3f}` | σ = `{sigma_cond_v2:.3f}`")

                if indep_v2 == "Sí — independientes":
                    st.success(
                        "✅ Ambas distribuciones son idénticas.  \n"
                        "Conocer $X_1$ no cambia nada sobre $X_2$."
                    )
                else:
                    st.warning(
                        f"⚠️ La distribución de $X_2$ **se desplaza** al conocer $X_1={x1_cond:.1f}$.  \n"
                        "Esto prueba que $X_1$ y $X_2$ **no son independientes**."
                    )

            with col_c2:
                y_range = np.linspace(-4, 4, 300)
                f2_marginal = stats.norm(0, 1).pdf(y_range)
                f2_cond = stats.norm(mu_cond_v2, sigma_cond_v2).pdf(y_range)

                fig_v2 = go.Figure()
                fig_v2.add_trace(go.Scatter(
                    x=y_range, y=f2_marginal,
                    mode='lines', name='f₂(x₂) — Marginal (sin info de X₁)',
                    line=dict(color='#e63946', width=3)
                ))
                fig_v2.add_trace(go.Scatter(
                    x=y_range, y=f2_cond,
                    mode='lines',
                    name=f'f₂|₁(x₂ | X₁={x1_cond:.1f}) — Condicional',
                    line=dict(color='#2196F3', width=3, dash='dash')
                ))
                fig_v2.add_vline(x=0, line_dash="dot",
                    line_color="#e63946", opacity=0.4,
                    annotation_text="μ₂ = 0", annotation_font_color="#e63946")
                if abs(mu_cond_v2) > 0.05:
                    fig_v2.add_vline(x=mu_cond_v2, line_dash="dot",
                        line_color="#2196F3", opacity=0.5,
                        annotation_text=f"E[X₂|x₁]={mu_cond_v2:.2f}",
                        annotation_font_color="#2196F3")
                fig_v2.update_layout(
                    xaxis_title="x₂", yaxis_title="Densidad",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    legend=dict(font=dict(color='white')),
                    height=400,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                fig_v2.update_xaxes(gridcolor='#333')
                fig_v2.update_yaxes(gridcolor='#333')
                st.plotly_chart(fig_v2, use_container_width=True)
                st.caption(
                    "Rojo sólido: distribución marginal de $X_2$ — lo que sabe sin información adicional.  "
                    "Azul punteado: distribución condicional dado $X_1 = x_1$.  "
                    "Cuando son independientes ambas curvas se superponen exactamente, "
                    "sin importar qué valor tome $x_1$."
                )

        # VIZ 3 — VERIFICANDO INDEPENDENCIA CON MUESTRAS
        with viz3:
            st.markdown(
                "**¿Cómo verificar si dos variables son independientes usando muestras?**  \n"
                "Si $X_1 \\perp X_2$, entonces $f_{1,2}(x_1,x_2) = f_1(x_1)\\cdot f_2(x_2)$.  \n"
                "Visualmente esto significa que la nube de puntos $(x_1, x_2)$ **no debe mostrar "
                "ninguna estructura o patrón**: los valores de $X_2$ no deben 'organizarse' "
                "según el valor de $X_1$.  \n\n"
                "Además, por la Propiedad (1): si son independientes, la distribución de $X_2$ "
                "dentro de cualquier franja vertical (un valor fijo de $X_1$) es siempre la misma."
            )
            col_v3a, col_v3b = st.columns([1, 2])
            with col_v3a:
                caso_v3 = st.radio(
                    "Escenario",
                    ["X₁ y X₂ independientes", "X₁ y X₂ dependientes"],
                    key="v3_caso"
                )
                n_v3 = st.slider("Número de observaciones", 200, 1500, 600, 100, key="v3_n")
                seed_v3 = st.slider("Semilla aleatoria", 0, 99, 42, key="v3_seed")
                franja = st.slider(
                    "Franja de X₁ a observar (±δ alrededor de 0)",
                    0.2, 1.5, 0.5, 0.1, key="v3_franja"
                )

                rng_v3 = np.random.default_rng(seed_v3)
                if caso_v3 == "X₁ y X₂ independientes":
                    X1_v3 = rng_v3.normal(0, 1, n_v3)
                    X2_v3 = rng_v3.normal(0, 1, n_v3)
                    msg_v3 = "✅ Independientes: la nube no muestra patrón."
                else:
                    Z1 = rng_v3.normal(0, 1, n_v3)
                    Z2 = rng_v3.normal(0, 1, n_v3)
                    X1_v3 = Z1
                    X2_v3 = 0.85*Z1 + np.sqrt(1-0.85**2)*Z2
                    msg_v3 = "❌ Dependientes: la nube muestra estructura lineal."

                mask = np.abs(X1_v3) < franja
                n_franja = mask.sum()
                media_fuera = np.mean(X2_v3[~mask]) if (~mask).sum() > 0 else 0
                media_dentro = np.mean(X2_v3[mask]) if mask.sum() > 0 else 0

                st.markdown("---")
                st.markdown(msg_v3)
                st.markdown(f"**Puntos en franja** |X₁| < {franja}: `{n_franja}`")
                st.markdown(f"**Media X₂ en franja:** `{media_dentro:.3f}`")
                st.markdown(f"**Media X₂ fuera:** `{media_fuera:.3f}`")
                if caso_v3 == "X₁ y X₂ independientes":
                    st.success(
                        "Las medias de X₂ son similares dentro y fuera de la franja:  \n"
                        "saber X₁ no cambia la distribución de X₂."
                    )
                else:
                    st.error(
                        "Las medias de X₂ difieren entre la franja y el resto:  \n"
                        "saber X₁ cambia lo que esperamos de X₂ — son dependientes."
                    )

            with col_v3b:
                colores_v3 = ['#4CAF50' if m else '#2196F3' for m in mask]
                fig_v3 = go.Figure()
                fig_v3.add_trace(go.Scatter(
                    x=X1_v3[~mask], y=X2_v3[~mask],
                    mode='markers',
                    marker=dict(color='#2196F3', size=4, opacity=0.4),
                    name='Fuera de franja'
                ))
                fig_v3.add_trace(go.Scatter(
                    x=X1_v3[mask], y=X2_v3[mask],
                    mode='markers',
                    marker=dict(color='#FF9800', size=5, opacity=0.9),
                    name=f'En franja |X₁|<{franja}'
                ))
                fig_v3.add_vline(x=franja, line_dash="dash",
                                 line_color="white", opacity=0.4)
                fig_v3.add_vline(x=-franja, line_dash="dash",
                                 line_color="white", opacity=0.4)
                fig_v3.add_hline(y=media_dentro, line_dash="dot",
                                 line_color="#FF9800", opacity=0.8,
                                 annotation_text=f"Media X₂|franja={media_dentro:.2f}",
                                 annotation_font_color="#FF9800")
                fig_v3.update_layout(
                    xaxis_title="X₁", yaxis_title="X₂",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=430,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white'))
                )
                fig_v3.update_xaxes(gridcolor='#222')
                fig_v3.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_v3, use_container_width=True)
                st.caption(
                    "Naranja: puntos dentro de la franja $|X_1| < \\delta$ (valor de $X_1$ conocido).  "
                    "Azul: resto de puntos.  "
                    "Si son independientes, la media de $X_2$ en la franja "
                    "debe ser similar a la del resto — la franja no aporta información."
                )

# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑAS 2–5 — PLACEHOLDER
# ═══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## Independencia de Vectores Aleatorios")
    st.markdown(
        "Extendemos el concepto de independencia al caso general de $N$ variables aleatorias, "
        "y también al caso en que **grupos** de variables son independientes entre sí como bloque."
    )

    teoria_v, visual_v = st.tabs(["📖 Teoría", "🎮 Visualización interactiva"])

    with teoria_v:

        # ── 0. MOTIVACIÓN ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">0 · ¿Por qué extender a N variables?</div>', unsafe_allow_html=True)
        st.markdown(
            "En finanzas rara vez trabajamos con dos activos. Un portafolio tiene $N$ activos, "
            "un modelo de riesgo crediticio puede involucrar decenas de variables. "
            "Necesitamos saber cuándo **todas** estas variables son mutuamente independientes, "
            "y cuándo podemos agruparlas en bloques independientes entre sí."
        )
        st.markdown(
            "La extensión no es trivial: que $X_1 \\perp X_2$ y $X_2 \\perp X_3$ "
            "**no implica** que $X_1 \\perp X_3$ ni que las tres sean mutuamente independientes. "
            "La independencia mutua es una condición más fuerte."
        )

        # ── 1. INDEPENDENCIA MUTUA DE N VARIABLES ─────────────────────────
        st.markdown('<div class="section-title">1 · Independencia mutua de N variables aleatorias</div>', unsafe_allow_html=True)

        st.markdown(
            "Cuando tenemos $N$ variables $X_1, X_2, \\ldots, X_N$, cada una genera eventos "
            "de la forma $A_{X_j} = \\{X_j \\in I_j\\}$. "
            "La independencia mutua exige que **todos** esos eventos sean simultáneamente independientes."
        )

        st.markdown('<div class="def-box"><span class="label-def">Definición — Independencia mutua</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $A_1, A_2, A_3$ eventos. Son **mutuamente independientes** si:"
        )
        st.latex(r"""
        \begin{aligned}
        &\text{(i)}\quad A_1 \perp A_2 \\
        &\text{(ii)}\quad A_1 \perp A_3 \\
        &\text{(iii)}\quad A_2 \perp A_3 \\
        &\text{(iv)}\quad \mathbb{P}(A_1 \cap A_2 \cap A_3) = \mathbb{P}(A_1)\,\mathbb{P}(A_2)\,\mathbb{P}(A_3)
        \end{aligned}
        """)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="obs-box"><span class="label-obs">Observación crítica</span>', unsafe_allow_html=True)
        st.markdown(
            "Las condiciones (i), (ii) y (iii) solos — independencia **por pares** — "
            "**no son suficientes** para garantizar la independencia mutua. "
            "Se necesita además (iv): que la probabilidad de la intersección de los tres sea "
            "el producto de las tres probabilidades individuales. "
            "Este es un error conceptual muy frecuente."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "En general, para $N$ eventos $A_1, \\ldots, A_N$, la independencia mutua requiere "
            "que **para todo subconjunto** $\\{i_1, \\ldots, i_k\\} \\subseteq \\{1, \\ldots, N\\}$:"
        )
        st.latex(r"\mathbb{P}(A_{i_1} \cap A_{i_2} \cap \cdots \cap A_{i_k}) = \mathbb{P}(A_{i_1})\,\mathbb{P}(A_{i_2})\cdots\mathbb{P}(A_{i_k})")

        # ── 2. CRITERIOS PRÁCTICOS PARA N VARIABLES ───────────────────────
        st.markdown('<div class="section-title">2 · Criterios prácticos para N variables</div>', unsafe_allow_html=True)

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema — Caso discreto (N variables)</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1, X_2, \\ldots, X_N$ variables aleatorias **discretas**. "
            "Son mutuamente independientes **si y solo si** para todo $(x_1, \\ldots, x_N)$:"
        )
        st.latex(r"p_{1,2,\ldots,N}(x_1,\ldots,x_N) = p_1(x_1)\,p_2(x_2)\cdots p_N(x_N)")
        st.markdown(
            "Es decir, la función de masa de probabilidad conjunta factoriza "
            "como el **producto de todas las marginales**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema — Caso continuo (N variables)</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1, X_2, \\ldots, X_N$ variables aleatorias **continuas** con densidad "
            "conjunta $f_{1,2,\\ldots,N}$ y marginales $f_j$. Son mutuamente independientes "
            "**si y solo si**:"
        )
        st.latex(r"f(x_1, x_2, \ldots, x_N) = f_1(x_1)\cdot f_2(x_2) \cdots f_N(x_N), \quad \forall\,(x_1,\ldots,x_N) \in \mathbb{R}^N")
        st.markdown(
            "donde $f_j$ es la **densidad marginal** de $X_j$, obtenida integrando sobre todas las demás variables."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📝 ¿Qué es exactamente la densidad marginal $f_j$?"):
            st.markdown(
                "La densidad marginal de $X_j$ se obtiene **integrando** la densidad conjunta "
                "sobre todos los demás valores:"
            )
            st.latex(r"f_j(x_j) = \int_{\mathbb{R}^{N-1}} f(x_1,\ldots,x_N)\,dx_1\cdots dx_{j-1}\,dx_{j+1}\cdots dx_N")
            st.markdown(
                "Es la distribución 'propia' de $X_j$, ignorando completamente el resto. "
                "Si la densidad conjunta factoriza, entonces cada marginal ya contiene toda "
                "la información de su variable — ninguna variable aporta información sobre las demás."
            )

        # ── 3. INDEPENDENCIA DE VECTORES ──────────────────────────────────
        st.markdown('<div class="section-title">3 · Independencia entre vectores aleatorios</div>', unsafe_allow_html=True)
        st.markdown(
            "La noción más general y poderosa: en lugar de pedir independencia variable a variable, "
            "pedimos independencia **bloque a bloque**. "
            "Esto aparece naturalmente cuando modelamos grupos de variables con estructura interna."
        )

        st.markdown('<div class="def-box"><span class="label-def">Definición — Independencia de vectores</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $(X_1, X_2, X_3)$ un vector aleatorio y $Y$ una variable aleatoria. "
            "Se dice que el vector $(X_1, X_2, X_3)$ y $Y$ son **independientes** si "
            "todos los eventos generados por $(X_1, X_2, X_3)$ y todos los generados por $Y$ "
            "son independientes."
        )
        st.markdown("Los eventos generados por $(X_1, X_2, X_3)$ son de la forma:")
        st.latex(r"A_{\mathbf{X}} = \{(X_1, X_2, X_3) \in I_1 \times I_2 \times I_3\}, \quad I_1, I_2, I_3 \subset \mathbb{R}")
        st.markdown("y los generados por $Y$:")
        st.latex(r"A_Y = \{Y \in I\}, \quad I \subset \mathbb{R}")
        st.markdown(
            "Son independientes si $\\mathbb{P}(A_{\\mathbf{X}} \\cap A_Y) = \\mathbb{P}(A_{\\mathbf{X}})\\cdot\\mathbb{P}(A_Y)$ "
            "para **todos** los posibles $I_1, I_2, I_3, I$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **En palabras:** conocer el valor del vector $(X_1, X_2, X_3)$ — "
            "incluyendo toda la estructura conjunta entre sus componentes — "
            "no aporta ninguna información sobre $Y$, y viceversa. "
            "El bloque completo es lo que se declara independiente de $Y$."
        )

        st.markdown('<div class="teo-box"><span class="label-teo">Criterio práctico — Independencia de vectores (continuo)</span>', unsafe_allow_html=True)
        st.markdown(
            "$(X_1, X_2, X_3)$ y $Y$ son independientes **si y solo si** "
            "la densidad conjunta factoriza entre el bloque y $Y$:"
        )
        st.latex(r"f(x_1, x_2, x_3, y) = f_{X_1,X_2,X_3}(x_1,x_2,x_3)\cdot f_Y(y)")
        st.markdown(
            "Nótese que dentro del bloque $(X_1, X_2, X_3)$, las variables pueden ser "
            "**dependientes entre sí** — lo que se requiere es que el bloque como unidad "
            "sea independiente de $Y$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 4. DETERMINACIÓN COMPLETA Y PARCIAL ───────────────────────────
        st.markdown('<div class="section-title">4 · Determinación entre un vector y una variable</div>', unsafe_allow_html=True)

        st.markdown('<div class="def-box"><span class="label-def">Definición — Determinación</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $(X_1, X_2, X_3)$ y $Y$ dependientes. "
            "Se dice que $(X_1, X_2, X_3)$ **determina completamente** a $Y$ si existe "
            "$g: \\mathbb{R}^3 \\to \\mathbb{R}$ tal que:"
        )
        st.latex(r"Y = g(X_1, X_2, X_3)")
        st.markdown(
            "Si no existe tal $g$, se dice que $(X_1, X_2, X_3)$ **determina parcialmente** a $Y$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        col_ej1, col_ej2 = st.columns(2)
        with col_ej1:
            st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo — Portafolio</span>', unsafe_allow_html=True)
            st.markdown(
                "Sean $X_1, X_2, X_3$ retornos de tres activos y $Y$ el retorno del portafolio:"
            )
            st.latex(r"Y = w_1 X_1 + w_2 X_2 + w_3 X_3 = g(X_1,X_2,X_3)")
            st.markdown(
                "$(X_1,X_2,X_3)$ determina **completamente** a $Y$: "
                "si conocemos los retornos de los tres activos, el retorno del portafolio "
                "queda determinado sin ambigüedad."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_ej2:
            st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo — Opción sobre índice</span>', unsafe_allow_html=True)
            st.markdown(
                "Sea $S = X_1 + X_2 + X_3$ el índice y $Y = \\max\\{S - K, 0\\}$ el payoff de un call."
            )
            st.latex(r"Y = \max\{X_1+X_2+X_3 - K,\, 0\} = g(X_1,X_2,X_3)")
            st.markdown(
                "$(X_1,X_2,X_3)$ determina **completamente** a $Y$. "
                "Pero $Y=0$ ocurre para infinitos valores posibles del índice ($S \\leq K$), "
                "así que $Y$ solo determina **parcialmente** al vector."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── 5. PROPIEDADES GENERALES ───────────────────────────────────────
        st.markdown('<div class="section-title">5 · Propiedades de la independencia mutua</div>', unsafe_allow_html=True)
        st.markdown('<div class="prop-box"><span class="label-prop">Propiedades</span>', unsafe_allow_html=True)
        st.markdown("Si $X_1, X_2, \\ldots, X_N$ son mutuamente independientes, entonces:")
        st.latex(r"\text{(1)}\quad E\left[\prod_{j=1}^N g_j(X_j)\right] = \prod_{j=1}^N E\bigl[g_j(X_j)\bigr]")
        st.latex(r"\text{(2)}\quad \text{Var}\left(\sum_{j=1}^N a_j X_j\right) = \sum_{j=1}^N a_j^2\,\text{Var}(X_j)")
        st.latex(r"\text{(3)}\quad \text{Cov}(X_i, X_j) = 0 \quad \forall\, i \neq j")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **(1)** La esperanza del producto factoriza en el producto de esperanzas — "
            "esto generaliza la propiedad (4) del caso $N=2$. "
            "**(2)** La varianza de una combinación lineal de variables independientes "
            "es simplemente la suma de las varianzas individuales ponderadas — "
            "**no hay términos de covarianza**. "
            "Esto es el fundamento matemático de la **diversificación** en finanzas. "
            "**(3)** Variables independientes son siempre no correlacionadas, "
            "aunque la recíproca no vale en general."
        )

        with st.expander("📐 Demostración de la propiedad (2)"):
            st.markdown("Sea $Y = \\sum_{j=1}^N a_j X_j$. Por definición de varianza:")
            st.latex(r"\text{Var}(Y) = E\left[\left(Y - EY\right)^2\right] = E\left[\left(\sum_j a_j(X_j - \mu_j)\right)^2\right]")
            st.markdown("Expandiendo el cuadrado:")
            st.latex(r"= \sum_j a_j^2\,E\left[(X_j-\mu_j)^2\right] + 2\sum_{i<j} a_i a_j\,E\left[(X_i-\mu_i)(X_j-\mu_j)\right]")
            st.markdown(
                "Por independencia mutua, $E[(X_i-\\mu_i)(X_j-\\mu_j)] = "
                "E[X_i-\\mu_i]\\cdot E[X_j-\\mu_j] = 0 \\cdot 0 = 0$ para $i \\neq j$. "
                "Los términos cruzados desaparecen y queda:"
            )
            st.latex(r"\text{Var}(Y) = \sum_{j=1}^N a_j^2\,\text{Var}(X_j) \qquad \blacksquare")

    # ── VISUALIZACIÓN ───────────────────────────────────────────────────────

    # ── VISUALIZACIÓN PESTAÑA 2 ─────────────────────────────────────────────
    with visual_v:
        st.markdown("### Herramientas visuales")
        vv1, vv2 = st.tabs([
            "🧩 Independencia mutua: ¿cómo se ve?",
            "📊 Varianza de una suma: términos diagonales"
        ])

        # VV1 — CÓMO SE VE LA INDEPENDENCIA MUTUA
        with vv1:
            st.markdown(
                "**Independencia mutua de $N$ variables:** $f(x_1,\\ldots,x_N) = f_1(x_1)\\cdots f_N(x_N)$.  \n"
                "En el caso $N=3$, esto implica que **todas** las proyecciones bivariadas "
                "$(X_1,X_2)$, $(X_1,X_3)$ y $(X_2,X_3)$ deben mostrar independencia.  \n\n"
                "Recuerda: que dos de ellas sean independientes **no garantiza** que las tres lo sean. "
                "Aquí puedes comparar los tres escenarios."
            )
            col_vv1a, col_vv1b = st.columns([1, 3])
            with col_vv1a:
                caso_vv1 = st.radio(
                    "Escenario",
                    [
                        "Las tres independientes",
                        "Solo X₁ y X₂ dependientes",
                        "Las tres dependientes"
                    ],
                    key="vv1_caso"
                )
                n_vv1 = st.slider("Observaciones", 200, 1000, 500, 100, key="vv1_n")
                seed_vv1 = st.slider("Semilla", 0, 99, 7, key="vv1_seed")
                rng_vv1 = np.random.default_rng(seed_vv1)

                if caso_vv1 == "Las tres independientes":
                    X1 = rng_vv1.normal(0,1,n_vv1)
                    X2 = rng_vv1.normal(0,1,n_vv1)
                    X3 = rng_vv1.normal(0,1,n_vv1)
                    veredicto = "✅ Las tres nubes no muestran patrón — mutuamente independientes."
                elif caso_vv1 == "Solo X₁ y X₂ dependientes":
                    Z1 = rng_vv1.normal(0,1,n_vv1)
                    Z2 = rng_vv1.normal(0,1,n_vv1)
                    X1 = Z1
                    X2 = 0.85*Z1 + np.sqrt(1-0.85**2)*Z2
                    X3 = rng_vv1.normal(0,1,n_vv1)
                    veredicto = "⚠️ X₁–X₂ muestra patrón (dependientes). X₁–X₃ y X₂–X₃ no."
                else:
                    Z1 = rng_vv1.normal(0,1,n_vv1)
                    Z2 = rng_vv1.normal(0,1,n_vv1)
                    Z3 = rng_vv1.normal(0,1,n_vv1)
                    X1 = Z1
                    X2 = 0.8*Z1 + np.sqrt(1-0.64)*Z2
                    X3 = 0.6*Z1 + 0.5*Z2 + np.sqrt(max(1-0.36-0.25,0))*Z3
                    veredicto = "❌ Las tres nubes muestran patrón — ningún par es independiente."

                st.markdown("---")
                st.info(veredicto)

            with col_vv1b:
                fig_vv1 = make_subplots(rows=1, cols=3,
                    subplot_titles=["X₁ vs X₂", "X₁ vs X₃", "X₂ vs X₃"])
                pares = [(X1,X2,"#2196F3"), (X1,X3,"#4CAF50"), (X2,X3,"#FF9800")]
                for idx, (xa, ya, col_p) in enumerate(pares, 1):
                    fig_vv1.add_trace(go.Scatter(
                        x=xa, y=ya, mode='markers',
                        marker=dict(color=col_p, size=3, opacity=0.4),
                        showlegend=False
                    ), row=1, col=idx)
                fig_vv1.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                for i in range(1,4):
                    fig_vv1.update_xaxes(gridcolor='#333', row=1, col=i)
                    fig_vv1.update_yaxes(gridcolor='#333', row=1, col=i)
                st.plotly_chart(fig_vv1, use_container_width=True)
                st.caption(
                    "Nube circular = sin patrón = indicativo de independencia en ese par.  "
                    "Nube elongada o con forma = hay dependencia.  "
                    "La independencia mutua requiere que las TRES nubes sean circulares."
                )

        # VV2 — VARIANZA DE UNA SUMA
        with vv2:
            st.markdown(
                "**Propiedad (2):** Si $X_1,\\ldots,X_N$ son mutuamente independientes:  \n"
            )
            st.latex(r"\text{Var}(a_1X_1+\cdots+a_NX_N) = a_1^2\text{Var}(X_1)+\cdots+a_N^2\text{Var}(X_N)")
            st.markdown(
                "No hay términos cruzados porque $\\text{Cov}(X_i,X_j)=0$ para todo $i\\neq j$.  \n"
                "Aquí verificamos esto numéricamente: generamos $N$ variables independientes, "
                "calculamos la varianza de su suma, y comparamos con la fórmula."
            )
            col_vv2a, col_vv2b = st.columns([1, 2])
            with col_vv2a:
                N_vv2 = st.slider("Número de variables N", 2, 6, 3, key="vv2_N")
                seed_vv2 = st.slider("Semilla", 0, 99, 10, key="vv2_seed")
                n_sim = 5000
                rng_vv2 = np.random.default_rng(seed_vv2)

                sigmas = [0.3 + 0.15*j for j in range(N_vv2)]
                pesos  = [1.0/N_vv2]*N_vv2

                muestras = [rng_vv2.normal(0, sigmas[j], n_sim) for j in range(N_vv2)]
                Y_sim = sum(pesos[j]*muestras[j] for j in range(N_vv2))

                var_formula = sum(pesos[j]**2 * sigmas[j]**2 for j in range(N_vv2))
                var_empirica = float(np.var(Y_sim))

                st.markdown("---")
                st.markdown("**Parámetros generados:**")
                for j in range(N_vv2):
                    st.markdown(f"$X_{j+1}$: σ = `{sigmas[j]:.2f}`, peso $a_{j+1}$ = `{pesos[j]:.3f}`")
                st.markdown("---")
                st.markdown(f"**Var(Y) por fórmula:** `{var_formula:.5f}`")
                st.markdown(f"**Var(Y) calculada:** `{var_empirica:.5f}`")
                if abs(var_formula - var_empirica) / var_formula < 0.05:
                    st.success("✅ Coinciden: la fórmula funciona porque son independientes.")

            with col_vv2b:
                labels_vv2 = [f"$a_{j+1}^2\\sigma_{j+1}^2$" for j in range(N_vv2)]
                values_vv2 = [pesos[j]**2 * sigmas[j]**2 for j in range(N_vv2)]
                colors_vv2 = ['#2196F3','#4CAF50','#FF9800','#e63946','#9C27B0','#00BCD4']

                fig_vv2 = go.Figure()
                fig_vv2.add_trace(go.Bar(
                    x=[f"X{j+1}" for j in range(N_vv2)],
                    y=values_vv2,
                    marker_color=colors_vv2[:N_vv2],
                    text=[f"{v:.4f}" for v in values_vv2],
                    textposition='outside',
                    textfont=dict(color='white'),
                    name="Contribución individual"
                ))
                fig_vv2.add_hline(
                    y=var_formula, line_dash="dash", line_color="white",
                    annotation_text=f"Var(Y) total = {var_formula:.4f}",
                    annotation_font_color="white"
                )
                fig_vv2.update_layout(
                    xaxis_title="Variable", yaxis_title="Contribución a Var(Y)",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    height=380,
                    margin=dict(l=0, r=0, t=20, b=0),
                    showlegend=False
                )
                fig_vv2.update_xaxes(gridcolor='#222')
                fig_vv2.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vv2, use_container_width=True)
                st.caption(
                    "Cada barra es la contribución $a_j^2\\sigma_j^2$ de esa variable a la varianza total.  "
                    "La línea punteada es Var(Y). Por independencia, la suma de las barras "
                    "es exactamente igual a la línea — sin ningún término cruzado."
                )
with tabs[2]:
    st.markdown("## Covarianza")
    st.markdown(
        "La covarianza es la primera herramienta que nos permite **cuantificar** "
        "la relación entre dos variables aleatorias: no solo detectar si existe dependencia, "
        "sino medir su **dirección** e **intensidad lineal**."
    )

    teoria_c, visual_c = st.tabs(["📖 Teoría", "🎮 Visualización interactiva"])

    with teoria_c:

        # ── 0. SETUP ───────────────────────────────────────────────────────
        st.markdown('<div class="section-title">0 · Configuración del espacio</div>', unsafe_allow_html=True)
        st.markdown(
            "Trabajamos en el espacio de probabilidad $(\\Omega, \\mathcal{F}, \\mathbb{P})$. "
            "Sean $(X_1, X_2)$ variables aleatorias con:"
        )
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.latex(r"\mu_j = \mu_{X_j} = E[X_j] \quad \text{(media)}")
            st.latex(r"\sigma_j^2 = \sigma_{X_j}^2 = \text{Var}(X_j) \quad \text{(varianza)}")
        with col_s2:
            st.latex(r"\sigma_j = \sigma_{X_j} = +\sqrt{\sigma_j^2} \quad \text{(desv. estándar)}")
            st.markdown(
                "Se exige $\\sigma_j > 0$, es decir, las variables deben tener "
                "varianza positiva — no son constantes."
            )

        # ── 1. DEFINICIÓN ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">1 · Definición de covarianza</div>', unsafe_allow_html=True)
        st.markdown(
            "La varianza de una sola variable mide cuánto se aleja $X$ de su media en promedio. "
            "La covarianza extiende esa idea a **dos** variables: mide cuánto se alejan "
            "$X_1$ y $X_2$ de sus medias **al mismo tiempo** y **en la misma dirección**."
        )
        st.markdown('<div class="def-box"><span class="label-def">Definición</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1$ y $X_2$ variables aleatorias. "
            "La **covarianza** de $X_1$ y $X_2$ se define como:"
        )
        st.latex(r"\text{Cov}(X_1, X_2) = E\bigl[(X_1 - \mu_1)(X_2 - \mu_2)\bigr]")
        st.markdown("En términos de la función $g(X_1, X_2) = (X_1-\\mu_1)(X_2-\\mu_2)$:")
        st.latex(r"\text{Cov}(X_1, X_2) = E\bigl[g(X_1, X_2)\bigr]")
        st.markdown("**Caso discreto:**")
        st.latex(r"\text{Cov}(X_1,X_2) = \sum_{x_2}\sum_{x_1}(x_1-\mu_1)(x_2-\mu_2)\,p_{1,2}(x_1,x_2)")
        st.markdown("**Caso continuo:**")
        st.latex(r"\text{Cov}(X_1,X_2) = \iint_{\mathbb{R}^2}(x_1-\mu_1)(x_2-\mu_2)\,f_{1,2}(x_1,x_2)\,dx_1\,dx_2")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── DISCUSIÓN DEL SIGNO ────────────────────────────────────────────
        st.markdown('<div class="section-title">2 · ¿Qué mide el signo de la covarianza?</div>', unsafe_allow_html=True)
        st.markdown(
            "El núcleo de la definición es el producto $(x_1-\\mu_1)(x_2-\\mu_2)$. "
            "Analicemos su signo caso a caso, ponderando por la densidad conjunta $f_{1,2}(x_1,x_2) > 0$:"
        )

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown('<div class="prop-box"><span class="label-prop">Cov > 0 → misma dirección</span>', unsafe_allow_html=True)
            st.markdown(
                "El producto $(x_1-\\mu_1)(x_2-\\mu_2) > 0$ cuando:"
            )
            st.latex(r"(+)(+): \; x_1 > \mu_1 \;\text{ y }\; x_2 > \mu_2")
            st.latex(r"(-)(-):\; x_1 < \mu_1 \;\text{ y }\; x_2 < \mu_2")
            st.markdown(
                "Valores **altos** de $X_1$ acompañados de valores **altos** de $X_2$, "
                "y valores **bajos** con **bajos**. "
                "Ponderando por $f_{1,2}$: estas configuraciones son las más probables. "
                "**$X_1$ y $X_2$ se mueven en la misma dirección.**"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_d2:
            st.markdown('<div class="obs-box"><span class="label-obs">Cov < 0 → dirección opuesta</span>', unsafe_allow_html=True)
            st.markdown(
                "El producto $(x_1-\\mu_1)(x_2-\\mu_2) < 0$ cuando:"
            )
            st.latex(r"(+)(-): \; x_1 > \mu_1 \;\text{ y }\; x_2 < \mu_2")
            st.latex(r"(-)(+): \; x_1 < \mu_1 \;\text{ y }\; x_2 > \mu_2")
            st.markdown(
                "Valores **altos** de $X_1$ acompañados de valores **bajos** de $X_2$, "
                "y viceversa. "
                "**$X_1$ y $X_2$ se mueven en dirección opuesta.**"
            )
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Cov = 0:** los pares que van en la misma dirección y los que van en dirección "
            "opuesta se compensan exactamente en el promedio ponderado. "
            "**No hay tendencia lineal** entre las variables."
        )

        # ── 3. PROPIEDADES ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">3 · Propiedades de la covarianza</div>', unsafe_allow_html=True)

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 1 — Relación con la varianza</span>', unsafe_allow_html=True)
        st.latex(r"\text{Cov}(X_1, X_1) = \text{Var}(X_1)")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración"):
            st.latex(r"\text{Cov}(X_1,X_1) = E[(X_1-\mu_1)(X_1-\mu_1)] = E[(X_1-\mu_1)^2] = \text{Var}(X_1) \quad \blacksquare")
        st.markdown(
            "> La varianza es un caso especial de covarianza: la covarianza de una variable consigo misma. "
            "Esto también justifica que $\\text{Var}(X_1) \\geq 0$."
        )

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 2 — Fórmula operacional</span>', unsafe_allow_html=True)
        st.latex(r"\text{Cov}(X_1, X_2) = E[X_1 X_2] - E[X_1]\cdot E[X_2]")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración"):
            st.latex(r"\text{Cov}(X_1,X_2) = E[(X_1-\mu_1)(X_2-\mu_2)]")
            st.latex(r"= E[X_1 X_2 - \mu_1 X_2 - \mu_2 X_1 + \mu_1\mu_2]")
            st.latex(r"= E[X_1 X_2] - \mu_1 \underbrace{E[X_2]}_{=\mu_2} - \mu_2\underbrace{E[X_1]}_{=\mu_1} + \mu_1\mu_2")
            st.latex(r"= E[X_1 X_2] - \mu_1\mu_2 - \mu_1\mu_2 + \mu_1\mu_2 = E[X_1 X_2] - \mu_1\mu_2 \quad \blacksquare")
        st.markdown(
            "> Esta fórmula es la más usada en la práctica. "
            "Permite calcular la covarianza sin restar la media en cada término — "
            "basta con $E[X_1 X_2]$ y las medias individuales."
        )

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 3 — Independencia implica covarianza cero</span>', unsafe_allow_html=True)
        st.markdown("Si $X_1$ y $X_2$ son **independientes**, entonces:")
        st.latex(r"\text{Cov}(X_1, X_2) = 0")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración"):
            st.markdown("Por independencia: $E[X_1 X_2] = E[X_1]\\cdot E[X_2]$. Entonces:")
            st.latex(r"\text{Cov}(X_1,X_2) = E[X_1 X_2] - E[X_1]E[X_2] = E[X_1]E[X_2] - E[X_1]E[X_2] = 0 \quad \blacksquare")

        st.markdown('<div class="obs-box"><span class="label-obs">⚠️ El recíproco es FALSO en general</span>', unsafe_allow_html=True)
        st.markdown(
            "**Covarianza cero NO implica independencia.** "
            "La covarianza solo captura asociación **lineal**. "
            "Puede existir una relación no lineal perfecta con $\\text{Cov}=0$. "
            "El contraejemplo clásico: sea $X \\sim N(0,1)$ y $Y = X^2$. "
            "Entonces $Y$ está completamente determinada por $X$, pero:"
        )
        st.latex(r"\text{Cov}(X, X^2) = E[X \cdot X^2] - E[X]\cdot E[X^2] = E[X^3] - 0 \cdot 1 = 0")
        st.markdown(
            "porque $E[X^3]=0$ por simetría de la normal. "
            "**$X$ y $X^2$ son perfectamente dependientes pero tienen covarianza cero.**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 4 — Transformaciones lineales afines</span>', unsafe_allow_html=True)
        st.markdown("Sean $Y_1 = \\alpha_1 + \\beta_1 X_1$ y $Y_2 = \\alpha_2 + \\beta_2 X_2$. Entonces:")
        st.latex(r"\text{Cov}(Y_1, Y_2) = \beta_1\,\beta_2\,\text{Cov}(X_1, X_2)")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración"):
            st.latex(r"\text{Cov}(Y_1,Y_2) = E[(Y_1-\mu_{Y_1})(Y_2-\mu_{Y_2})]")
            st.latex(r"= E[(\alpha_1+\beta_1 X_1 - \alpha_1-\beta_1\mu_1)(\alpha_2+\beta_2 X_2 - \alpha_2-\beta_2\mu_2)]")
            st.latex(r"= E[\beta_1(X_1-\mu_1)\cdot\beta_2(X_2-\mu_2)] = \beta_1\beta_2\,E[(X_1-\mu_1)(X_2-\mu_2)]")
            st.latex(r"= \beta_1\beta_2\,\text{Cov}(X_1,X_2) \quad \blacksquare")
        st.markdown(
            "> Las constantes aditivas $\\alpha_1, \\alpha_2$ no afectan la covarianza — "
            "solo desplazan la media. Los factores multiplicativos $\\beta_1, \\beta_2$ sí la escalan. "
            "**La covarianza no es invariante a escala** — este problema lo resolverá la correlación."
        )

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 5 — Varianza de una suma</span>', unsafe_allow_html=True)
        st.latex(r"\text{Var}(X_1 + X_2) = \text{Var}(X_1) + \text{Var}(X_2) + 2\,\text{Cov}(X_1,X_2)")
        st.markdown("Más generalmente, para $Y = \\sum_{j=1}^N a_j X_j$:")
        st.latex(r"\text{Var}(Y) = \sum_{j=1}^N a_j^2\,\text{Var}(X_j) + 2\sum_{i<j} a_i a_j\,\text{Cov}(X_i,X_j)")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración (caso N=2)"):
            st.latex(r"\text{Var}(X_1+X_2) = E[(X_1+X_2-\mu_1-\mu_2)^2]")
            st.latex(r"= E[((X_1-\mu_1)+(X_2-\mu_2))^2]")
            st.latex(r"= E[(X_1-\mu_1)^2] + E[(X_2-\mu_2)^2] + 2E[(X_1-\mu_1)(X_2-\mu_2)]")
            st.latex(r"= \text{Var}(X_1) + \text{Var}(X_2) + 2\,\text{Cov}(X_1,X_2) \quad \blacksquare")

        st.markdown(
            "> 💡 Esta propiedad es **central en la teoría de portafolios**. "
            "El riesgo de un portafolio no es la suma de los riesgos individuales — "
            "hay términos de covarianza cruzada. Si $\\text{Cov}<0$, el riesgo se reduce."
        )

        # ── 4. EJEMPLO FINANCIERO ──────────────────────────────────────────
        st.markdown('<div class="section-title">4 · Aplicación financiera</div>', unsafe_allow_html=True)

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo — Profit de call y put sobre el mismo subyacente</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $S$: precio del subyacente. Definimos:"
        )
        st.latex(r"X_1 = c - \max\{S-K,\,0\} \quad \text{(profit del call corto)}")
        st.latex(r"X_2 = \max\{S-K,\,0\} - c \quad \text{(profit del call largo)}")
        st.markdown("Entonces $X_2 = -X_1$, es decir $X_2 = \\alpha + \\beta X_1$ con $\\alpha=0$, $\\beta=-1$.")
        st.markdown("Por la Propiedad 4:")
        st.latex(r"\text{Cov}(X_1,X_2) = (-1)\cdot\text{Cov}(X_1,X_1) = -\text{Var}(X_1) < 0")
        st.markdown(
            "**Interpretación:** cuando el call largo gana, el call corto pierde exactamente lo mismo. "
            "Son perfectamente contrapuestos — máxima covarianza negativa posible."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo — ¿Qué signo tiene Cov(S, max{S−K, 0})?</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $S$: precio del subyacente, $X = \\max\\{S-K, 0\\}$: payoff del call.  \n"
            "Cuando $S$ sube por encima de $\\mu_S$, el payoff $X$ también tiende a subir "
            "(si $S > K$, $X = S-K$ crece). "
            "Cuando $S$ baja, $X$ cae o se queda en 0.  \n"
            "En ambos casos $(S-\\mu_S)$ y $(X-\\mu_X)$ tienden a tener el **mismo signo**.  \n"
            "Por tanto, $\\text{Cov}(S, X) > 0$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── VISUALIZACIÓN ───────────────────────────────────────────────────────

    # ── VISUALIZACIÓN PESTAÑA 3 — COVARIANZA ────────────────────────────────
    with visual_c:
        st.markdown("### Herramientas visuales")
        vc1, vc2 = st.tabs([
            "📐 El signo de la covarianza",
            "🧮 Fórmula operacional: verificación"
        ])

        # VC1 — SIGNO DE LA COVARIANZA
        with vc1:
            st.markdown(
                "**La covarianza es el promedio ponderado del producto $(x_1-\\mu_1)(x_2-\\mu_2)$.**  \n"
                "El signo depende de en qué cuadrantes (respecto a $(\\mu_1,\\mu_2)$) "
                "se concentra la mayoría de la masa de probabilidad.  \n\n"
                "Los puntos **verdes** contribuyen positivamente a la covarianza: "
                "$(x_1-\\mu_1)$ y $(x_2-\\mu_2)$ tienen el mismo signo.  \n"
                "Los puntos **rojos** contribuyen negativamente: signos opuestos."
            )
            col_vc1a, col_vc1b = st.columns([1, 2])
            with col_vc1a:
                tipo_rel = st.select_slider(
                    "Tipo de relación entre X₁ y X₂",
                    options=["Fuerte negativa", "Negativa", "Ninguna", "Positiva", "Fuerte positiva"],
                    value="Ninguna", key="vc1_tipo"
                )
                rho_map_vc1 = {
                    "Fuerte negativa": -0.9,
                    "Negativa": -0.6,
                    "Ninguna": 0.0,
                    "Positiva": 0.6,
                    "Fuerte positiva": 0.9
                }
                rho_vc1 = rho_map_vc1[tipo_rel]
                n_vc1 = st.slider("Observaciones", 200, 800, 400, 100, key="vc1_n")
                seed_vc1 = st.slider("Semilla", 0, 99, 7, key="vc1_seed")

                rng_vc1 = np.random.default_rng(seed_vc1)
                pts = rng_vc1.multivariate_normal([0,0], [[1,rho_vc1],[rho_vc1,1]], n_vc1)
                x1_pts, x2_pts = pts[:,0], pts[:,1]
                prod = x1_pts * x2_pts  # μ₁=μ₂=0
                cov_emp = float(np.mean(prod))
                pct_pos = float(np.mean(prod > 0))*100

                st.markdown("---")
                st.markdown(f"**Cov calculada:** `{cov_emp:.3f}`")
                st.markdown(f"**% puntos que contribuyen positivamente:** `{pct_pos:.1f}%`")
                st.markdown(f"**% puntos que contribuyen negativamente:** `{100-pct_pos:.1f}%`")

                if cov_emp > 0.05:
                    st.success("Cov > 0: X₁ y X₂ se mueven en la misma dirección.")
                elif cov_emp < -0.05:
                    st.error("Cov < 0: X₁ y X₂ se mueven en dirección opuesta.")
                else:
                    st.info("Cov ≈ 0: no hay tendencia lineal predominante.")

            with col_vc1b:
                colores_vc1 = ['#4CAF50' if p > 0 else '#e63946' for p in prod]
                fig_vc1 = go.Figure()
                fig_vc1.add_trace(go.Scatter(
                    x=x1_pts, y=x2_pts, mode='markers',
                    marker=dict(color=colores_vc1, size=5, opacity=0.6),
                    showlegend=False
                ))
                fig_vc1.add_vline(x=0, line_dash="dash", line_color="white", opacity=0.5,
                                  annotation_text="μ₁=0", annotation_font_color="white")
                fig_vc1.add_hline(y=0, line_dash="dash", line_color="white", opacity=0.5,
                                  annotation_text="μ₂=0", annotation_font_color="white")
                for txt, xpos, ypos in [
                    ("(+,+) ✅", 1.8, 2.5), ("(−,−) ✅", -2.8, -2.5),
                    ("(+,−) ❌", 1.8, -2.5), ("(−,+) ❌", -2.8, 2.5)
                ]:
                    fig_vc1.add_annotation(x=xpos, y=ypos, text=txt,
                        showarrow=False, font=dict(size=10, color='white'), opacity=0.5)
                fig_vc1.update_layout(
                    xaxis_title="x₁", yaxis_title="x₂",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=420,
                    margin=dict(l=0, r=0, t=20, b=0)
                )
                fig_vc1.update_xaxes(gridcolor='#222', range=[-3.5,3.5])
                fig_vc1.update_yaxes(gridcolor='#222', range=[-3.5,3.5])
                st.plotly_chart(fig_vc1, use_container_width=True)
                st.caption(
                    "Verde ✅: el punto contribuye positivamente a la covarianza "
                    "— mismo signo en ambas desviaciones.  "
                    "Rojo ❌: contribuye negativamente — signos opuestos.  "
                    "El signo de Cov depende de qué color domina."
                )

        # VC2 — FÓRMULA OPERACIONAL
        with vc2:
            st.markdown(
                "**Propiedad 2 — Fórmula operacional:**  \n"
            )
            st.latex(r"\text{Cov}(X_1,X_2) = E[X_1 X_2] - E[X_1]\cdot E[X_2]")
            st.markdown(
                "Aquí verificamos numéricamente que calcular la covarianza por la definición directa "
                "y por la fórmula operacional da el mismo resultado.  \n"
                "También observamos que $E[X_1 X_2] \\neq E[X_1]\\cdot E[X_2]$ cuando hay dependencia, "
                "y que se igualan cuando las variables son independientes."
            )
            col_vc2a, col_vc2b = st.columns([1,2])
            with col_vc2a:
                mu1_vc2 = st.slider("μ₁", -2.0, 2.0, 1.0, 0.5, key="vc2_m1")
                mu2_vc2 = st.slider("μ₂", -2.0, 2.0, 2.0, 0.5, key="vc2_m2")
                s1_vc2  = st.slider("σ₁", 0.3, 2.0, 1.0, 0.1, key="vc2_s1")
                s2_vc2  = st.slider("σ₂", 0.3, 2.0, 1.0, 0.1, key="vc2_s2")
                cov_vc2_param = st.slider(
                    "Cov(X₁,X₂)",
                    float(-s1_vc2*s2_vc2*0.95),
                    float(s1_vc2*s2_vc2*0.95),
                    float(s1_vc2*s2_vc2*0.6),
                    float(s1_vc2*s2_vc2*0.05),
                    key="vc2_cov"
                )
                n_vc2 = 3000
                rng_vc2 = np.random.default_rng(21)
                C_vc2 = [[s1_vc2**2, cov_vc2_param],[cov_vc2_param, s2_vc2**2]]
                smp = rng_vc2.multivariate_normal([mu1_vc2, mu2_vc2], C_vc2, n_vc2)
                x1s, x2s = smp[:,0], smp[:,1]

                EX1  = float(np.mean(x1s))
                EX2  = float(np.mean(x2s))
                EX1X2= float(np.mean(x1s*x2s))
                cov_def = float(np.mean((x1s-EX1)*(x2s-EX2)))
                cov_op  = EX1X2 - EX1*EX2

                st.markdown("---")
                st.markdown(f"**E[X₁]:** `{EX1:.3f}`")
                st.markdown(f"**E[X₂]:** `{EX2:.3f}`")
                st.markdown(f"**E[X₁·X₂]:** `{EX1X2:.3f}`")
                st.markdown(f"**E[X₁]·E[X₂]:** `{EX1*EX2:.3f}`")
                st.markdown("---")
                st.markdown(f"**Cov (definición):** `{cov_def:.4f}`")
                st.markdown(f"**Cov (fórmula op.):** `{cov_op:.4f}`")
                st.markdown(f"**Cov objetivo:** `{cov_vc2_param:.4f}`")
                if abs(cov_def - cov_op) < 0.01:
                    st.success("✅ Ambas fórmulas coinciden.")

            with col_vc2b:
                fig_vc2 = go.Figure()
                fig_vc2.add_trace(go.Scatter(
                    x=x1s, y=x2s, mode='markers',
                    marker=dict(color='#2196F3', size=3, opacity=0.25),
                    name='Observaciones'
                ))
                fig_vc2.add_vline(x=EX1, line_dash="dash", line_color="#e63946",
                                  annotation_text=f"E[X₁]={EX1:.2f}",
                                  annotation_font_color="#e63946")
                fig_vc2.add_hline(y=EX2, line_dash="dash", line_color="#4CAF50",
                                  annotation_text=f"E[X₂]={EX2:.2f}",
                                  annotation_font_color="#4CAF50")
                fig_vc2.update_layout(
                    xaxis_title="x₁", yaxis_title="x₂",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=420,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white'))
                )
                fig_vc2.update_xaxes(gridcolor='#222')
                fig_vc2.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vc2, use_container_width=True)
                st.caption(
                    "La diferencia $E[X_1X_2] - E[X_1]E[X_2]$ es exactamente la covarianza.  "
                    "Cuando la nube no tiene patrón (Cov≈0), "
                    "$E[X_1X_2] \\approx E[X_1]\\cdot E[X_2]$."
                )
with tabs[3]:
    st.markdown("## Correlación")
    st.markdown(
        "La covarianza mide la dirección de la relación lineal entre dos variables, "
        "pero su magnitud depende de las unidades de medida. "
        "La correlación resuelve ese problema: es una versión **estandarizada** de la covarianza, "
        "sin unidades, acotada entre $-1$ y $1$."
    )

    teoria_r, visual_r = st.tabs(["📖 Teoría", "🎮 Visualización interactiva"])

    with teoria_r:

        # ── 1. DEFINICIÓN ──────────────────────────────────────────────────
        st.markdown('<div class="section-title">1 · Definición del coeficiente de correlación</div>', unsafe_allow_html=True)
        st.markdown(
            "El problema de la covarianza es que su magnitud no es interpretable directamente: "
            "$\\text{Cov}=0.5$ puede ser mucho o poco dependiendo de las escalas de $X_1$ y $X_2$. "
            "Si multiplicamos $X_1$ por 1000 (e.g., cambiar de miles a unidades), "
            "la covarianza se multiplica por 1000 — pero la relación entre las variables no cambió. "
            "Necesitamos una medida **adimensional**."
        )

        st.markdown('<div class="def-box"><span class="label-def">Definición — Coeficiente de correlación de Pearson</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1$ y $X_2$ variables aleatorias con $\\sigma_1 > 0$ y $\\sigma_2 > 0$. "
            "El **coeficiente de correlación** (de Pearson) se define como:"
        )
        st.latex(r"\rho = \rho_{1,2} = \rho_{X_1,X_2} = \frac{\text{Cov}(X_1,X_2)}{\sigma_1 \cdot \sigma_2}")
        st.markdown("Expandiendo con la definición de covarianza:")
        st.latex(r"\rho = \frac{E[(X_1-\mu_1)(X_2-\mu_2)]}{\sqrt{E[(X_1-\mu_1)^2]}\;\sqrt{E[(X_2-\mu_2)^2]}}")
        st.markdown(
            "**$\\rho$ es adimensional** — al dividir por $\\sigma_1\\sigma_2$ se cancelan las unidades. "
            "Nótese que $\\sigma_1\\sigma_2 > 0$ siempre (por hipótesis), por lo que la definición es válida."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **En palabras:** $\\rho$ es la covarianza pero calculada sobre las versiones "
            "estandarizadas $Z_1 = (X_1-\\mu_1)/\\sigma_1$ y $Z_2 = (X_2-\\mu_2)/\\sigma_2$. "
            "Al estandarizar, eliminamos la escala y solo queda la estructura de dependencia lineal."
        )

        # ── 2. RELACIÓN CON COVARIANZA ─────────────────────────────────────
        st.markdown('<div class="section-title">2 · Relación entre correlación y covarianza</div>', unsafe_allow_html=True)
        st.markdown('<div class="prop-box"><span class="label-prop">Observación — El signo de ρ coincide con el de la covarianza</span>', unsafe_allow_html=True)
        st.markdown("Como $\\sigma_1\\cdot\\sigma_2 > 0$, dividir por él no cambia el signo:")
        st.latex(r"\text{Cov}(X_1,X_2) > 0 \;\Longleftrightarrow\; \rho > 0")
        st.latex(r"\text{Cov}(X_1,X_2) < 0 \;\Longleftrightarrow\; \rho < 0")
        st.latex(r"\text{Cov}(X_1,X_2) = 0 \;\Longleftrightarrow\; \rho = 0")
        st.markdown(
            "Por tanto, el signo de $\\rho$ tiene la misma interpretación que el de la covarianza: "
            "positivo = misma dirección, negativo = dirección opuesta, cero = sin tendencia lineal."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 3. PROPIEDADES ─────────────────────────────────────────────────
        st.markdown('<div class="section-title">3 · Propiedades del coeficiente de correlación</div>', unsafe_allow_html=True)

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 1 — Invarianza bajo transformaciones lineales</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $Y_1 = aX_1$ y $Y_2 = bX_2$ con $ab \\neq 0$. Entonces:"
        )
        st.latex(r"\rho(Y_1, Y_2) = \rho(X_1, X_2)")
        st.markdown(
            "La correlación **no cambia** al reescalar las variables. "
            "Esto es precisamente lo que la hace superior a la covarianza como medida de relación."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración"):
            st.markdown("Por la Propiedad 4 de covarianza: $\\text{Cov}(aX_1, bX_2) = ab\\,\\text{Cov}(X_1,X_2)$.")
            st.markdown("Las desviaciones estándar se transforman como $\\sigma_{Y_1} = |a|\\sigma_1$ y $\\sigma_{Y_2} = |b|\\sigma_2$. Entonces:")
            st.latex(r"\rho(Y_1,Y_2) = \frac{ab\,\text{Cov}(X_1,X_2)}{|a|\sigma_1 \cdot |b|\sigma_2} = \frac{ab}{|ab|}\cdot\rho(X_1,X_2)")
            st.markdown(
                "Si $ab > 0$: $\\rho(Y_1,Y_2) = \\rho(X_1,X_2)$.  \n"
                "Si $ab < 0$: $\\rho(Y_1,Y_2) = -\\rho(X_1,X_2)$.  \n"
                "Para el enunciado con $Y_1 = aX_1$, $Y_2 = bX_2$ con $ab \\neq 0$, "
                "el signo puede cambiar si $ab < 0$ — lo que se preserva es el **valor absoluto**. "
                "Cuando $a, b > 0$ la correlación es exactamente igual. $\\blacksquare$"
            )

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 2 — Acotamiento</span>', unsafe_allow_html=True)
        st.latex(r"-1 \leq \rho \leq 1")
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración — Desigualdad de Cauchy-Schwarz"):
            st.markdown(
                "Sea $t \\in \\mathbb{R}$. Consideremos la variable aleatoria "
                "$W = (X_1 - \\mu_1) + t(X_2-\\mu_2)$. Como $W^2 \\geq 0$:"
            )
            st.latex(r"0 \leq E[W^2] = \text{Var}(X_1) + 2t\,\text{Cov}(X_1,X_2) + t^2\text{Var}(X_2)")
            st.latex(r"= \sigma_1^2 + 2t\,\text{Cov}(X_1,X_2) + t^2\sigma_2^2")
            st.markdown(
                "Este es un polinomio en $t$ que siempre es $\\geq 0$, "
                "por lo que su discriminante debe ser $\\leq 0$:"
            )
            st.latex(r"(2\,\text{Cov}(X_1,X_2))^2 - 4\sigma_1^2\sigma_2^2 \leq 0")
            st.latex(r"\Rightarrow \text{Cov}(X_1,X_2)^2 \leq \sigma_1^2\sigma_2^2")
            st.latex(r"\Rightarrow \left(\frac{\text{Cov}(X_1,X_2)}{\sigma_1\sigma_2}\right)^2 \leq 1 \;\Rightarrow\; |\rho| \leq 1 \quad \blacksquare")

        st.markdown('<div class="prop-box"><span class="label-prop">Propiedad 3 — Relación lineal perfecta</span>', unsafe_allow_html=True)
        st.markdown("Sea $X_2 = \\alpha + \\beta X_1$ con $\\alpha \\in \\mathbb{R}$, $\\beta \\neq 0$. Entonces:")
        st.latex(r"\beta > 0 \;\Rightarrow\; \rho = +1 \qquad \beta < 0 \;\Rightarrow\; \rho = -1")
        st.markdown(
            "Los extremos $\\rho = \\pm 1$ se alcanzan **si y solo si** existe una relación lineal exacta "
            "entre las variables. Cualquier relación no lineal — por perfecta que sea — "
            "produce $|\\rho| < 1$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración (caso β > 0)"):
            st.markdown("Si $X_2 = \\alpha + \\beta X_1$ con $\\beta > 0$:")
            st.latex(r"\text{Cov}(X_1, X_2) = \text{Cov}(X_1,\, \alpha+\beta X_1) = \beta\,\text{Cov}(X_1,X_1) = \beta\,\sigma_1^2")
            st.latex(r"\sigma_{X_2} = \sqrt{\text{Var}(\alpha+\beta X_1)} = \beta\,\sigma_1")
            st.latex(r"\rho = \frac{\beta\sigma_1^2}{\sigma_1 \cdot \beta\sigma_1} = \frac{\beta\sigma_1^2}{\beta\sigma_1^2} = 1 \quad \blacksquare")

        # ── 4. INTERPRETACIÓN ──────────────────────────────────────────────
        st.markdown('<div class="section-title">4 · Interpretación de ρ</div>', unsafe_allow_html=True)

        col_int1, col_int2 = st.columns(2)
        with col_int1:
            st.markdown('<div class="def-box"><span class="label-def">Valores extremos</span>', unsafe_allow_html=True)
            st.latex(r"\rho = +1: \text{ relación lineal positiva perfecta}")
            st.latex(r"\rho = -1: \text{ relación lineal negativa perfecta}")
            st.markdown(
                "En ambos casos, con saber el valor de $X_1$ podemos predecir $X_2$ **exactamente** "
                "usando la ecuación de la recta."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_int2:
            st.markdown('<div class="obs-box"><span class="label-obs">Valores intermedios</span>', unsafe_allow_html=True)
            st.latex(r"0 < \rho < 1: \text{ tendencia lineal positiva (imperfecta)}")
            st.latex(r"-1 < \rho < 0: \text{ tendencia lineal negativa (imperfecta)}")
            st.latex(r"\rho = 0: \text{ sin tendencia lineal}")
            st.markdown(
                "Cuanto más cerca de $\\pm 1$, más fuerte la relación lineal. "
                "Cuanto más cerca de $0$, más débil."
            )
            st.markdown("</div>", unsafe_allow_html=True)

        # ── 5. LA TRAMPA CRÍTICA ───────────────────────────────────────────
        st.markdown('<div class="section-title">5 · La trampa crítica: ρ = 0 no implica independencia</div>', unsafe_allow_html=True)

        st.markdown('<div class="obs-box"><span class="label-obs">⚠️ Error conceptual más común</span>', unsafe_allow_html=True)
        st.markdown(
            "La correlación **solo mide relación lineal**. "
            "Una correlación de cero significa que no hay tendencia lineal — "
            "pero puede existir una dependencia no lineal perfecta. "
        )
        st.markdown("**Contraejemplo clásico:** sea $X \\sim N(0,1)$ y $Y = X^2$.")
        st.latex(r"Y = X^2 \;\Rightarrow\; Y \text{ está completamente determinada por } X")
        st.latex(r"\text{Cov}(X,Y) = E[X \cdot X^2] - E[X]\cdot E[X^2] = E[X^3] - 0 \cdot 1 = 0")
        st.markdown(
            "porque $E[X^3] = 0$ por simetría de la distribución normal. Entonces $\\rho = 0$.  \n"
            "Sin embargo, $X$ e $Y = X^2$ son **completamente dependientes**."
        )
        st.markdown("**Resumen de implicaciones:**")
        st.latex(r"X_1 \perp X_2 \;\Rightarrow\; \rho = 0 \quad \text{(siempre verdadero)}")
        st.latex(r"\rho = 0 \;\not\Rightarrow\; X_1 \perp X_2 \quad \text{(falso en general)}")
        st.latex(r"\rho = 0 \;\Leftrightarrow\; X_1 \perp X_2 \quad \text{(verdadero SOLO bajo normalidad conjunta)}")
        st.markdown("</div>", unsafe_allow_html=True)

        # ── 6. APLICACIONES FINANCIERAS ────────────────────────────────────
        st.markdown('<div class="section-title">6 · Aplicaciones financieras</div>', unsafe_allow_html=True)

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo 1 — Correlación entre retornos de activos</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1, X_2$ los retornos de dos acciones. "
            "La varianza del portafolio $Y = w_1 X_1 + w_2 X_2$ es:"
        )
        st.latex(r"\text{Var}(Y) = w_1^2\sigma_1^2 + w_2^2\sigma_2^2 + 2w_1 w_2\,\underbrace{\rho\,\sigma_1\sigma_2}_{\text{Cov}(X_1,X_2)}")
        st.markdown(
            "La correlación aparece directamente en la fórmula de riesgo. "
            "Activos con $\\rho < 0$ se **diversifican** mutuamente — "
            "la volatilidad del portafolio es menor que la de cada activo individual. "
            "Con $\\rho = 1$ no hay diversificación posible."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo 2 — Correlación entre precio y payoff del call</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $S$: precio del subyacente, $X = \\max\\{S-K, 0\\}$: payoff del call.  \n"
            "Ya vimos que $\\text{Cov}(S,X) > 0$. Por tanto $\\rho(S,X) > 0$.  \n"
            "¿Es $\\rho = 1$? No, porque la relación no es lineal: cuando $S \\leq K$, $X = 0$ "
            "independientemente de cuánto varíe $S$. La relación es **no lineal** $\\Rightarrow$ $\\rho < 1$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejemplo 3 — Call corto y call largo</span>', unsafe_allow_html=True)
        st.markdown(
            "Sea $X_1 = c - \\max\\{S-K,0\\}$ (call corto) y $X_2 = \\max\\{S-K,0\\} - c$ (call largo).  \n"
            "Tenemos $X_2 = -X_1$, es decir $\\beta = -1$. Por la Propiedad 3:"
        )
        st.latex(r"\rho(X_1, X_2) = -1")
        st.markdown(
            "Relación lineal negativa perfecta: cada sol que gana el comprador "
            "es un sol que pierde el vendedor."
        )
        st.markdown("</div>", unsafe_allow_html=True)

    # ── VISUALIZACIÓN ───────────────────────────────────────────────────────

    # ── VISUALIZACIÓN PESTAÑA 4 — CORRELACIÓN ──────────────────────────────
    with visual_r:
        st.markdown("### Herramientas visuales")
        vr1, vr2 = st.tabs([
            "📊 ρ como medida de relación lineal",
            "⚠️ ρ = 0 no implica independencia"
        ])

        # VR1 — ρ COMO MEDIDA
        with vr1:
            st.markdown(
                "**¿Qué aspecto tiene una nube de puntos para distintos valores de $\\rho$?**  \n"
                "La correlación $\\rho = \\text{Cov}(X_1,X_2)/(\\sigma_1\\sigma_2)$ es adimensional "
                "y mide la fuerza de la relación lineal.  \n\n"
                "Observa cómo cambia la nube de puntos al variar $\\rho$, "
                "y nota que cambiar las escalas $\\sigma_1, \\sigma_2$ **no altera** el valor de $\\rho$ "
                "— esa es su ventaja sobre la covarianza."
            )
            col_vr1a, col_vr1b = st.columns([1,2])
            with col_vr1a:
                rho_vr1 = st.slider("Correlación ρ", -1.0, 1.0, 0.0, 0.05, key="vr1_rho")
                s1_vr1  = st.slider("σ₁ (escala de X₁)", 0.5, 3.0, 1.0, 0.1, key="vr1_s1")
                s2_vr1  = st.slider("σ₂ (escala de X₂)", 0.5, 3.0, 1.5, 0.1, key="vr1_s2")
                n_vr1   = st.slider("Observaciones", 200, 800, 400, 100, key="vr1_n")
                seed_vr1 = st.slider("Semilla", 0, 99, 0, key="vr1_seed")

                rng_vr1 = np.random.default_rng(seed_vr1)
                cov_vr1 = rho_vr1*s1_vr1*s2_vr1
                smp_vr1 = rng_vr1.multivariate_normal(
                    [0,0],
                    [[s1_vr1**2, cov_vr1],[cov_vr1, s2_vr1**2]],
                    n_vr1
                )
                x1_vr1, x2_vr1 = smp_vr1[:,0], smp_vr1[:,1]
                rho_emp = float(np.corrcoef(x1_vr1, x2_vr1)[0,1])
                cov_emp_vr1 = float(np.cov(x1_vr1, x2_vr1)[0,1])

                st.markdown("---")
                st.markdown(f"**ρ fijado:** `{rho_vr1:.2f}`")
                st.markdown(f"**ρ calculado:** `{rho_emp:.3f}`")
                st.markdown(f"**Cov calculada:** `{cov_emp_vr1:.3f}`")
                st.markdown(f"**σ₁·σ₂:** `{s1_vr1*s2_vr1:.3f}`")
                st.markdown(f"**Cov / (σ₁·σ₂) = ρ:** `{cov_emp_vr1/(s1_vr1*s2_vr1):.3f}`")
                st.markdown(
                    "> Nota: aunque cambies σ₁ y σ₂, "
                    "la última fórmula siempre da ρ."
                )

            with col_vr1b:
                fig_vr1 = go.Figure()
                fig_vr1.add_trace(go.Scatter(
                    x=x1_vr1, y=x2_vr1, mode='markers',
                    marker=dict(color='#2196F3', size=4, opacity=0.5),
                    name='Observaciones'
                ))
                if abs(rho_vr1) > 0.05:
                    m_vr1 = cov_emp_vr1 / float(np.var(x1_vr1))
                    b_vr1 = float(np.mean(x2_vr1)) - m_vr1*float(np.mean(x1_vr1))
                    xl = np.linspace(x1_vr1.min(), x1_vr1.max(), 50)
                    fig_vr1.add_trace(go.Scatter(
                        x=xl, y=m_vr1*xl+b_vr1, mode='lines',
                        line=dict(color='#e63946', width=2, dash='dash'),
                        name='Tendencia lineal'
                    ))
                fig_vr1.update_layout(
                    xaxis_title="x₁", yaxis_title="x₂",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=420,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white'))
                )
                fig_vr1.update_xaxes(gridcolor='#222')
                fig_vr1.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vr1, use_container_width=True)
                st.caption(
                    "ρ≈0: nube circular, sin tendencia.  "
                    "ρ>0: nube inclinada hacia arriba — X₁ y X₂ se mueven juntos.  "
                    "ρ<0: nube inclinada hacia abajo.  "
                    "ρ=±1: todos los puntos sobre una recta — relación lineal perfecta."
                )

        # VR2 — ρ=0 NO IMPLICA INDEPENDENCIA
        with vr2:
            st.markdown(
                "**El contraejemplo más importante del curso:**  \n"
                "Que $\\rho = 0$ no implica independencia.  \n\n"
                "La correlación solo mide relación **lineal**. "
                "Puede existir una dependencia funcional perfecta y sin embargo $\\rho = 0$.  \n\n"
                "**Contraejemplo:** sea $X \\sim N(0,1)$ y $Y = X^2$."
            )
            st.latex(r"\text{Cov}(X,X^2) = E[X^3] - E[X]\cdot E[X^2] = 0 - 0\cdot 1 = 0 \;\Rightarrow\; \rho = 0")
            st.markdown(
                "Sin embargo, $Y$ está **completamente determinada** por $X$: "
                "dado $X=x$, el valor $Y=x^2$ es exacto. "
                "Son perfectamente dependientes pero con correlación cero."
            )

            col_vr2a, col_vr2b = st.columns([1,2])
            with col_vr2a:
                n_vr2 = st.slider("Observaciones", 300, 2000, 800, 100, key="vr2_n")
                seed_vr2 = st.slider("Semilla", 0, 99, 5, key="vr2_seed")
                ver_lineal = st.checkbox(
                    "Comparar con relación lineal (ρ≠0)",
                    False, key="vr2_lin"
                )

                rng_vr2 = np.random.default_rng(seed_vr2)
                X_vr2 = rng_vr2.normal(0, 1, n_vr2)
                Y_vr2 = X_vr2**2
                rho_cuad = float(np.corrcoef(X_vr2, Y_vr2)[0,1])

                if ver_lineal:
                    Z_lin = X_vr2 + rng_vr2.normal(0, 0.4, n_vr2)
                    rho_lin = float(np.corrcoef(X_vr2, Z_lin)[0,1])

                st.markdown("---")
                st.markdown(f"**ρ(X, X²) = ** `{rho_cuad:.4f}` ≈ 0")
                st.error(
                    f"Y=X² está completamente determinada por X, "
                    f"pero ρ ≈ {rho_cuad:.3f} ≈ 0.  \n"
                    "**La correlación no detecta la dependencia no lineal.**"
                )
                if ver_lineal:
                    st.markdown(f"**ρ(X, X+ruido) =** `{rho_lin:.4f}`")
                    st.success("La relación lineal sí es captada por ρ.")

                st.markdown("---")
                st.markdown("**Resumen:**")
                st.latex(r"X_1 \perp X_2 \Rightarrow \rho = 0 \quad \text{(siempre)}")
                st.latex(r"\rho = 0 \not\Rightarrow X_1 \perp X_2 \quad \text{(en general)}")

            with col_vr2b:
                fig_vr2 = go.Figure()
                fig_vr2.add_trace(go.Scatter(
                    x=X_vr2, y=Y_vr2, mode='markers',
                    marker=dict(color='#FF9800', size=4, opacity=0.5),
                    name=f'(X, X²)  ρ={rho_cuad:.3f}'
                ))
                xc = np.linspace(-3.2, 3.2, 200)
                fig_vr2.add_trace(go.Scatter(
                    x=xc, y=xc**2, mode='lines',
                    line=dict(color='#e63946', width=2),
                    name='Y = X² (curva exacta)'
                ))
                if ver_lineal:
                    fig_vr2.add_trace(go.Scatter(
                        x=X_vr2, y=Z_lin, mode='markers',
                        marker=dict(color='#4CAF50', size=4, opacity=0.4),
                        name=f'(X, X+ruido)  ρ={rho_lin:.3f}'
                    ))
                fig_vr2.update_layout(
                    xaxis_title="X", yaxis_title="Y",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=420,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white'))
                )
                fig_vr2.update_xaxes(gridcolor='#222')
                fig_vr2.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vr2, use_container_width=True)
                st.caption(
                    "Naranja: puntos (X, X²). La curva roja muestra la dependencia exacta.  "
                    "La nube sigue perfectamente la parábola, pero ρ≈0 "
                    "porque la relación es simétrica — la correlación no la detecta."
                )
with tabs[4]:
    st.markdown("## Distribución Normal Multivariada")
    st.markdown(
        "La distribución normal multivariada es la extensión natural de la distribución normal "
        "al caso de $N$ variables aleatorias conjuntas. "
        "Es el objeto central de la teoría de portafolios y modelos de riesgo financiero."
    )

    teoria_n, visual_n = st.tabs(["📖 Teoría", "🎮 Visualización interactiva"])

    with teoria_n:

        # ── 0. PUNTO DE PARTIDA ────────────────────────────────────────────
        st.markdown('<div class="section-title">0 · Punto de partida: la normal univariada</div>', unsafe_allow_html=True)
        st.markdown(
            "Antes de construir la normal multivariada, recordamos cómo se construye "
            "la normal univariada a partir de la estándar."
        )
        st.markdown('<div class="def-box"><span class="label-def">Recordatorio</span>', unsafe_allow_html=True)
        st.markdown("Sea $Z \\sim N(0,1)$ con densidad:")
        st.latex(r"f(z) = \frac{1}{\sqrt{2\pi}}\,e^{-\frac{1}{2}z^2}, \quad z \in \mathbb{R}")
        st.markdown("Si definimos $X = \\mu + \\sigma Z$ con $\\sigma > 0$, entonces $X \\sim N(\\mu, \\sigma^2)$ con densidad:")
        st.latex(r"f(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\,e^{-\frac{1}{2}\frac{(x-\mu)^2}{\sigma^2}}, \quad x \in \mathbb{R}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "> 💡 La idea clave es que **toda normal se construye como transformación lineal** "
            "de una normal estándar. Esta idea se extiende al caso multivariado."
        )

        # ── 1. CONSTRUCCIÓN — CASO INDEPENDIENTE ──────────────────────────
        st.markdown('<div class="section-title">1 · Construcción: caso independiente (N variables)</div>', unsafe_allow_html=True)
        st.markdown(
            "El primer paso es entender el caso más simple: $N$ normales estándar "
            "**independientes**. ¿Cuál es su densidad conjunta?"
        )

        st.markdown('<div class="ejemplo-box"><span class="label-ej">Ejercicio fundacional</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $Z_1, Z_2, \\ldots, Z_N$ variables aleatorias independientes con $Z_j \\sim N(0,1)$. "
            "Hallar la densidad conjunta de $(Z_1, Z_2, \\ldots, Z_N)$."
        )
        st.markdown("**Solución:** por independencia, la densidad conjunta factoriza:")
        st.latex(r"f(z_1,\ldots,z_N) = f_1(z_1)\cdot f_2(z_2)\cdots f_N(z_N)")
        st.latex(r"= \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z_1^2} \cdot \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z_2^2} \cdots \frac{1}{\sqrt{2\pi}}e^{-\frac{1}{2}z_N^2}")
        st.latex(r"= \left(\frac{1}{\sqrt{2\pi}}\right)^N \exp\!\left\{-\frac{1}{2}\sum_{j=1}^N z_j^2\right\}, \quad (z_1,\ldots,z_N) \in \mathbb{R}^N")
        st.markdown("</div>", unsafe_allow_html=True)
        st.markdown(
            "Esta es la **densidad normal estándar $N$-variada**: los exponentes se suman "
            "porque los factores se multiplican. "
            "En notación matricial, $\\sum_j z_j^2 = \\mathbf{z}'\\mathbf{z}$ donde $\\mathbf{z} = (z_1,\\ldots,z_N)'$."
        )

        # ── 2. CONSTRUCCIÓN GENERAL — N=2 CON CORRELACIÓN ─────────────────
        st.markdown('<div class="section-title">2 · Construcción general: caso N=2 con correlación</div>', unsafe_allow_html=True)
        st.markdown(
            "Ahora introducimos dependencia. La idea es construir $X_1, X_2$ como "
            "**transformaciones lineales** de normales estándar independientes, "
            "de modo que tengan la correlación deseada."
        )

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema — Construcción bivariada</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $Z_1, Z_2$ independientes con $Z_j \\sim N(0,1)$. "
            "Sean $\\mu_1, \\mu_2 \\in \\mathbb{R}$, $\\sigma_1, \\sigma_2 > 0$ y $\\rho \\in (-1,1)$. "
            "Definimos:"
        )
        st.latex(r"""
        \begin{cases}
        X_1 = \mu_1 + \sigma_1 Z_1 \\[6pt]
        X_2 = \mu_2 + \sigma_2\rho\, Z_1 + \sigma_2\sqrt{1-\rho^2}\, Z_2
        \end{cases}
        """)
        st.markdown("Entonces $(X_1, X_2)$ tiene distribución normal bivariada con:")
        st.latex(r"X_j \sim N(\mu_j, \sigma_j^2), \quad \text{Cov}(X_1,X_2) = \rho\sigma_1\sigma_2")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **¿Por qué esta construcción funciona?** "
            "$X_2$ tiene dos componentes: una parte proporcional a $Z_1$ (que comparte con $X_1$, "
            "generando la dependencia) y una parte proporcional a $Z_2$ (independiente de $X_1$, "
            "generando la variabilidad propia). "
            "El factor $\\sqrt{1-\\rho^2}$ garantiza que $\\text{Var}(X_2) = \\sigma_2^2$."
        )

        with st.expander("📐 Verificación de que Cov(X₁, X₂) = ρσ₁σ₂"):
            st.latex(r"\text{Cov}(X_1, X_2) = \text{Cov}(\mu_1+\sigma_1 Z_1,\; \mu_2+\sigma_2\rho Z_1+\sigma_2\sqrt{1-\rho^2}Z_2)")
            st.latex(r"= \text{Cov}(\sigma_1 Z_1,\; \sigma_2\rho Z_1+\sigma_2\sqrt{1-\rho^2}Z_2)")
            st.latex(r"= \sigma_1\sigma_2\rho\,\text{Cov}(Z_1,Z_1) + \sigma_1\sigma_2\sqrt{1-\rho^2}\,\underbrace{\text{Cov}(Z_1,Z_2)}_{=0\text{ (indep.)}}")
            st.latex(r"= \sigma_1\sigma_2\rho\cdot\text{Var}(Z_1) = \sigma_1\sigma_2\rho \cdot 1 = \rho\sigma_1\sigma_2 \quad \blacksquare")

        st.markdown("**En notación matricial:** $\\mathbf{X} = \\boldsymbol{\\mu} + Q\\mathbf{Z}$, donde:")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            st.latex(r"Q = \begin{pmatrix}\sigma_1 & 0 \\ \sigma_2\rho & \sigma_2\sqrt{1-\rho^2}\end{pmatrix}")
            st.markdown("$Q$ es la **matriz de Cholesky** de $C$.")
        with col_m2:
            st.latex(r"C = Q'Q = \begin{pmatrix}\sigma_1^2 & \rho\sigma_1\sigma_2 \\ \rho\sigma_1\sigma_2 & \sigma_2^2\end{pmatrix}")
            st.markdown("$C$ es la **matriz de covarianza** de $(X_1,X_2)$.")

        st.markdown(
            "La matriz $C$ recoge toda la información de varianzas y covarianzas: "
            "$c_{11} = \\text{Var}(X_1)$, $c_{22} = \\text{Var}(X_2)$, $c_{12} = \\text{Cov}(X_1,X_2) = \\rho\\sigma_1\\sigma_2$."
        )

        # ── 3. DENSIDAD CONJUNTA ───────────────────────────────────────────
        st.markdown('<div class="section-title">3 · La densidad conjunta bivariada</div>', unsafe_allow_html=True)
        st.markdown(
            "A partir de la construcción anterior, podemos hallar la densidad conjunta de $(X_1,X_2)$ "
            "mediante el método del jacobiano. El resultado es:"
        )

        st.markdown('<div class="def-box"><span class="label-def">Densidad normal bivariada</span>', unsafe_allow_html=True)
        st.latex(r"""
        f(x_1,x_2) = \left(\frac{1}{\sqrt{2\pi}}\right)^2 \frac{1}{\det Q}
        \exp\!\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})'\,C^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\},
        \quad (x_1,x_2)\in\mathbb{R}^2
        """)
        st.markdown("donde $\\det Q = \\sigma_1\\sigma_2\\sqrt{1-\\rho^2}$ y:")
        st.latex(r"(\mathbf{x}-\boldsymbol{\mu})'\,C^{-1}(\mathbf{x}-\boldsymbol{\mu}) = \frac{1}{1-\rho^2}\left[\frac{(x_1-\mu_1)^2}{\sigma_1^2} - \frac{2\rho(x_1-\mu_1)(x_2-\mu_2)}{\sigma_1\sigma_2} + \frac{(x_2-\mu_2)^2}{\sigma_2^2}\right]")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Forma cuadrática:** la expresión $-\\frac{1}{2}(\\mathbf{x}-\\boldsymbol{\\mu})'C^{-1}(\\mathbf{x}-\\boldsymbol{\\mu})$ "
            "es el análogo multivariado de $-\\frac{(x-\\mu)^2}{2\\sigma^2}$ en la normal univariada. "
            "Las curvas de nivel donde esta expresión es constante son **elipses** centradas en $(\\mu_1, \\mu_2)$."
        )

        # ── 4. DEFINICIÓN FORMAL GENERAL ──────────────────────────────────
        st.markdown('<div class="section-title">4 · Definición formal — caso general N variables</div>', unsafe_allow_html=True)

        st.markdown('<div class="def-box"><span class="label-def">Definición — Distribución normal multivariada</span>', unsafe_allow_html=True)
        st.markdown(
            "Se dice que $(X_1, X_2, \\ldots, X_N)$ tiene **distribución normal multivariada** "
            "con parámetros $\\boldsymbol{\\mu} = (\\mu_1,\\ldots,\\mu_N)$ y matriz de covarianza $C$ "
            "(definida positiva) si su función de densidad conjunta es:"
        )
        st.latex(r"f(\mathbf{x}) = \left(\frac{1}{\sqrt{2\pi}}\right)^N \frac{1}{\det Q}\exp\!\left\{-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})'\,C^{-1}(\mathbf{x}-\boldsymbol{\mu})\right\}, \quad \mathbf{x} \in \mathbb{R}^N")
        st.markdown(
            "donde $C = Q'Q$ ($Q$ es la **matriz de Cholesky** de $C$, triangular superior "
            "con elementos positivos en la diagonal), y el vector $\\mathbf{X}$ se construye como:"
        )
        st.latex(r"\mathbf{X} = \boldsymbol{\mu} + Q\mathbf{Z}, \quad \mathbf{Z} = (Z_1,\ldots,Z_N)' \text{ i.i.d. } N(0,1)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "La matriz de covarianza $C$ tiene estructura:"
        )
        st.latex(r"C = \begin{pmatrix}c_{11} & c_{12} & \cdots & c_{1N} \\ c_{12} & c_{22} & \cdots & c_{2N} \\ \vdots & & \ddots & \vdots \\ c_{1N} & \cdots & & c_{NN}\end{pmatrix}")
        st.markdown("donde $c_{jj} = \\text{Var}(X_j)$ y $c_{ij} = \\text{Cov}(X_i, X_j)$ para $i \\neq j$.")

        # ── 5. TEOREMAS IMPORTANTES ────────────────────────────────────────
        st.markdown('<div class="section-title">5 · Teoremas importantes</div>', unsafe_allow_html=True)

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema 1 — Las marginales son normales</span>', unsafe_allow_html=True)
        st.markdown(
            "Si $(X_1,\\ldots,X_N)$ tiene distribución normal multivariada, entonces:"
        )
        st.latex(r"X_j \sim N(\mu_j,\, c_{jj}), \quad j = 1,2,\ldots,N")
        st.markdown(
            "Cada variable marginal es normal con media $\\mu_j$ y varianza $c_{jj}$."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> ⚠️ **La recíproca es falsa:** que las marginales sean normales no garantiza "
            "que la distribución conjunta sea normal multivariada. "
            "Se puede construir una distribución conjunta no normal cuyas marginales son normales."
        )

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema 2 — Independencia ⟺ Covarianza cero (bajo normalidad)</span>', unsafe_allow_html=True)
        st.markdown(
            "Si $(X_1,\\ldots,X_N)$ tiene distribución normal multivariada y $C$ es **diagonal**, "
            "entonces $X_1,\\ldots,X_N$ son **mutuamente independientes**."
        )
        st.latex(r"C = \begin{pmatrix}\sigma_1^2 & 0 & \cdots & 0 \\ 0 & \sigma_2^2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & \cdots & & \sigma_N^2\end{pmatrix} \;\Rightarrow\; X_1 \perp X_2 \perp \cdots \perp X_N")
        st.markdown(
            "Bajo normalidad conjunta: $\\text{Cov}(X_i,X_j) = 0 \\Leftrightarrow X_i \\perp X_j$. "
            "**Este es el único caso general donde covarianza cero implica independencia.**"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("📐 Demostración (caso N=2)"):
            st.markdown("Si $C$ es diagonal: $\\rho = 0$, $Q = \\text{diag}(\\sigma_1, \\sigma_2)$, $C^{-1} = \\text{diag}(1/\\sigma_1^2, 1/\\sigma_2^2)$.")
            st.latex(r"f(x_1,x_2) = \frac{1}{2\pi\sigma_1\sigma_2}\exp\!\left\{-\frac{1}{2}\left[\frac{(x_1-\mu_1)^2}{\sigma_1^2}+\frac{(x_2-\mu_2)^2}{\sigma_2^2}\right]\right\}")
            st.latex(r"= \left(\frac{1}{\sqrt{2\pi}\sigma_1}e^{-\frac{(x_1-\mu_1)^2}{2\sigma_1^2}}\right)\!\left(\frac{1}{\sqrt{2\pi}\sigma_2}e^{-\frac{(x_2-\mu_2)^2}{2\sigma_2^2}}\right) = f_1(x_1)\cdot f_2(x_2)")
            st.markdown("La densidad conjunta factoriza $\\Rightarrow$ $X_1 \\perp X_2$. $\\blacksquare$")

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema 3 — La condicional es normal</span>', unsafe_allow_html=True)
        st.markdown(
            "Si $(X_1, X_2)$ tiene distribución normal bivariada, entonces:"
        )
        st.latex(r"X_2 \mid X_1 = x_1 \;\sim\; N\!\left(\mu_2 + \sigma_2\rho\,\frac{x_1-\mu_1}{\sigma_1},\;\; \sigma_2^2(1-\rho^2)\right)")
        st.markdown("Es decir:")
        col_cond1, col_cond2 = st.columns(2)
        with col_cond1:
            st.latex(r"E[X_2 \mid X_1=x_1] = \mu_2 + \frac{\sigma_2}{\sigma_1}\rho\,(x_1-\mu_1)")
        with col_cond2:
            st.latex(r"\text{Var}(X_2 \mid X_1=x_1) = \sigma_2^2(1-\rho^2)")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Interpretación:**  \n"
            "> - La **media condicional** es lineal en $x_1$ — si $\\rho > 0$ y $x_1 > \\mu_1$, "
            "esperamos que $X_2$ también esté por encima de su media.  \n"
            "> - La **varianza condicional** es $\\sigma_2^2(1-\\rho^2)$ — siempre menor que $\\sigma_2^2$. "
            "Cuanto mayor $|\\rho|$, más reduce la varianza conocer $X_1$. "
            "Si $|\\rho|=1$, la varianza condicional es 0: $X_2$ queda determinado exactamente.  \n"
            "> - Si $\\rho = 0$: la condicional es igual a la marginal — $X_1$ no aporta información sobre $X_2$."
        )

        with st.expander("📐 Demostración"):
            st.markdown(
                "De la construcción $X_2 = \\mu_2 + \\sigma_2\\rho Z_1 + \\sigma_2\\sqrt{1-\\rho^2}Z_2$ "
                "y $X_1 = \\mu_1 + \\sigma_1 Z_1$, despejamos $Z_1 = (X_1-\\mu_1)/\\sigma_1$. "
                "Condicionando en $X_1 = x_1$:"
            )
            st.latex(r"E[X_2 \mid X_1=x_1] = \mu_2 + \sigma_2\rho\underbrace{E[Z_1\mid X_1=x_1]}_{=(x_1-\mu_1)/\sigma_1} + \sigma_2\sqrt{1-\rho^2}\underbrace{E[Z_2\mid X_1=x_1]}_{=0\text{ (indep.)}}")
            st.latex(r"= \mu_2 + \frac{\sigma_2\rho}{\sigma_1}(x_1-\mu_1)")
            st.latex(r"\text{Var}(X_2\mid X_1=x_1) = \sigma_2^2(1-\rho^2)\underbrace{\text{Var}(Z_2\mid X_1=x_1)}_{=1} = \sigma_2^2(1-\rho^2) \quad \blacksquare")

        st.markdown('<div class="teo-box"><span class="label-teo">Teorema 4 — Combinaciones lineales son normales</span>', unsafe_allow_html=True)
        st.markdown(
            "Si $(X_1,\\ldots,X_N)$ tiene distribución normal multivariada y "
            "$Y = b + \\sum_{j=1}^N a_j X_j$ con $b, a_j \\in \\mathbb{R}$, entonces:"
        )
        st.latex(r"Y \sim N(EY,\, \text{Var}\,Y)")
        st.markdown("donde:")
        st.latex(r"EY = b + \sum_{j=1}^N a_j\mu_j")
        st.latex(r"\text{Var}\,Y = \mathbf{a}'\,C\,\mathbf{a} = \sum_{j=1}^N a_j^2\,c_{jj} + 2\sum_{i<j}a_i a_j\,c_{ij}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Aplicación directa:** el retorno de cualquier portafolio formado por "
            "activos con distribución normal multivariada es también **normal**. "
            "Esto justifica el uso del modelo normal en la teoría de portafolios de Markowitz."
        )

    # ── VISUALIZACIÓN ───────────────────────────────────────────────────────

    # ── VISUALIZACIÓN PESTAÑA 5 — NORMAL MULTIVARIADA ──────────────────────
    with visual_n:
        st.markdown("### Herramientas visuales")
        vn1, vn2, vn3 = st.tabs([
            "🏔️ Densidad bivariada y sus parámetros",
            "📊 Distribución condicional",
            "📐 Combinaciones lineales son normales"
        ])

        # VN1 — DENSIDAD BIVARIADA
        with vn1:
            st.markdown(
                "**La densidad normal bivariada como superficie en $\\mathbb{R}^3$.**  \n"
                "Los parámetros son $\\mu_1, \\mu_2$ (medias), $\\sigma_1, \\sigma_2$ (desv. estándar) "
                "y $\\rho$ (correlación).  \n\n"
                "- $\\mu_1, \\mu_2$ desplazan el centro de la campana.  \n"
                "- $\\sigma_1, \\sigma_2$ controlan el ancho en cada dirección.  \n"
                "- $\\rho$ inclina la campana: si $\\rho=0$ la superficie es simétrica, "
                "si $\\rho \\neq 0$ se tuerce hacia la diagonal."
            )
            col_vn1a, col_vn1b = st.columns([1, 2])
            with col_vn1a:
                mu1_vn1 = st.slider("μ₁", -2.0, 2.0, 0.0, 0.5, key="vn1_m1")
                mu2_vn1 = st.slider("μ₂", -2.0, 2.0, 0.0, 0.5, key="vn1_m2")
                s1_vn1  = st.slider("σ₁", 0.3, 2.0, 1.0, 0.1, key="vn1_s1")
                s2_vn1  = st.slider("σ₂", 0.3, 2.0, 1.0, 0.1, key="vn1_s2")
                rho_vn1 = st.slider("ρ", -0.95, 0.95, 0.0, 0.05, key="vn1_rho")

                cov_vn1 = rho_vn1*s1_vn1*s2_vn1
                C_vn1 = [[s1_vn1**2, cov_vn1],[cov_vn1, s2_vn1**2]]
                st.markdown("---")
                st.markdown("**Matriz de covarianza C:**")
                st.latex(
                    r"C = \begin{pmatrix}"
                    + f"{s1_vn1**2:.2f} & {cov_vn1:.2f}"
                    + r"\\ "
                    + f"{cov_vn1:.2f} & {s2_vn1**2:.2f}"
                    + r"\end{pmatrix}"
                )
                if abs(rho_vn1) < 0.05:
                    st.success("ρ≈0: C es diagonal → X₁ y X₂ son independientes.")
                elif rho_vn1 > 0:
                    st.info(f"ρ={rho_vn1:.2f}: la campana se inclina hacia (+,+) y (−,−).")
                else:
                    st.info(f"ρ={rho_vn1:.2f}: la campana se inclina hacia (+,−) y (−,+).")

            with col_vn1b:
                x_r = np.linspace(mu1_vn1-3.5*s1_vn1, mu1_vn1+3.5*s1_vn1, 70)
                y_r = np.linspace(mu2_vn1-3.5*s2_vn1, mu2_vn1+3.5*s2_vn1, 70)
                X_g, Y_g = np.meshgrid(x_r, y_r)
                pos = np.dstack((X_g, Y_g))
                rv = stats.multivariate_normal(mean=[mu1_vn1,mu2_vn1], cov=C_vn1)
                Z_g = rv.pdf(pos)

                fig_vn1 = go.Figure(go.Surface(
                    x=x_r, y=y_r, z=Z_g,
                    colorscale='Viridis', opacity=0.9, showscale=False
                ))
                fig_vn1.update_layout(
                    scene=dict(
                        xaxis_title="x₁", yaxis_title="x₂", zaxis_title="f(x₁,x₂)",
                        bgcolor='rgba(0,0,0,0)',
                        camera=dict(eye=dict(x=1.5, y=-1.5, z=1.2))
                    ),
                    height=450,
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                st.plotly_chart(fig_vn1, use_container_width=True)
                st.caption(
                    "La 'campana' se centra en (μ₁, μ₂). "
                    "Con ρ=0 es simétrica respecto a los ejes. "
                    "Con ρ≠0 se tuerce — la masa se concentra en la diagonal "
                    "correspondiente al signo de ρ."
                )

        # VN2 — DISTRIBUCIÓN CONDICIONAL
        with vn2:
            st.markdown(
                "**Teorema 3 — La condicional es normal:**  \n"
            )
            st.latex(
                r"X_2 \mid X_1=x_1 \;\sim\; N\!\left("
                r"\mu_2 + \frac{\sigma_2}{\sigma_1}\rho\,(x_1-\mu_1),\;\;"
                r"\sigma_2^2(1-\rho^2)\right)"
            )
            st.markdown(
                "Dos efectos al condicionar:  \n"
                "1. La **media condicional** se desplaza desde $\\mu_2$ en función de cuánto "
                "se aleja $x_1$ de $\\mu_1$ y del signo de $\\rho$.  \n"
                "2. La **varianza condicional** $\\sigma_2^2(1-\\rho^2)$ siempre es menor que $\\sigma_2^2$ "
                "— conocer $X_1$ reduce la incertidumbre sobre $X_2$.  \n\n"
                "Si $\\rho=0$: la condicional es igual a la marginal — $X_1$ no informa sobre $X_2$.  \n"
                "Si $|\\rho|=1$: la varianza condicional es 0 — $X_2$ queda determinado exactamente."
            )
            col_vn2a, col_vn2b = st.columns([1,2])
            with col_vn2a:
                mu1_vn2 = st.slider("μ₁", -2.0, 2.0, 0.0, 0.5, key="vn2_m1")
                mu2_vn2 = st.slider("μ₂", -2.0, 2.0, 0.0, 0.5, key="vn2_m2")
                s1_vn2  = st.slider("σ₁", 0.3, 2.0, 1.0, 0.1, key="vn2_s1")
                s2_vn2  = st.slider("σ₂", 0.3, 2.0, 1.0, 0.1, key="vn2_s2")
                rho_vn2 = st.slider("ρ", -0.95, 0.95, 0.7, 0.05, key="vn2_rho")
                x1_c = st.slider(
                    "Valor condicionado x₁",
                    float(mu1_vn2 - 3*s1_vn2),
                    float(mu1_vn2 + 3*s1_vn2),
                    float(mu1_vn2 + s1_vn2),
                    float(s1_vn2*0.1),
                    key="vn2_x1"
                )

                mu_c = mu2_vn2 + (s2_vn2/s1_vn2)*rho_vn2*(x1_c - mu1_vn2)
                var_c = s2_vn2**2*(1 - rho_vn2**2)
                sig_c = np.sqrt(var_c)
                reduccion = (1 - (1-rho_vn2**2))*100

                st.markdown("---")
                st.markdown("**Marginal** $X_2 \\sim N(\\mu_2, \\sigma_2^2)$:")
                st.markdown(f"Media = `{mu2_vn2:.3f}` | σ = `{s2_vn2:.3f}`")
                st.markdown(f"**Condicional** $X_2 \\mid X_1={x1_c:.2f}$:")
                st.markdown(f"Media = `{mu_c:.3f}` | σ = `{sig_c:.3f}`")
                st.markdown(f"**Reducción de varianza:** `{reduccion:.1f}%`")

                if abs(rho_vn2) < 0.05:
                    st.info("ρ≈0: la condicional coincide con la marginal.")
                else:
                    st.success(
                        f"Conocer X₁={x1_c:.2f} reduce la varianza de X₂ en {reduccion:.1f}%."
                    )

            with col_vn2b:
                y_p = np.linspace(mu2_vn2 - 4*s2_vn2, mu2_vn2 + 4*s2_vn2, 400)
                f_m = stats.norm(mu2_vn2, s2_vn2).pdf(y_p)
                f_c = stats.norm(mu_c, sig_c).pdf(y_p)

                fig_vn2 = go.Figure()
                fig_vn2.add_trace(go.Scatter(
                    x=y_p, y=f_m, mode='lines',
                    name='Marginal f₂(x₂)',
                    line=dict(color='#e63946', width=3)
                ))
                fig_vn2.add_trace(go.Scatter(
                    x=y_p, y=f_c, mode='lines',
                    name=f'Condicional f₂|₁(x₂|X₁={x1_c:.2f})',
                    line=dict(color='#2196F3', width=3, dash='dash')
                ))
                fig_vn2.add_vline(x=mu2_vn2, line_dash="dot",
                    line_color="#e63946", opacity=0.5,
                    annotation_text=f"μ₂={mu2_vn2:.2f}",
                    annotation_font_color="#e63946")
                fig_vn2.add_vline(x=mu_c, line_dash="dot",
                    line_color="#2196F3", opacity=0.6,
                    annotation_text=f"E[X₂|x₁]={mu_c:.2f}",
                    annotation_font_color="#2196F3")
                fig_vn2.update_layout(
                    xaxis_title="x₂", yaxis_title="Densidad",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white'))
                )
                fig_vn2.update_xaxes(gridcolor='#222')
                fig_vn2.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vn2, use_container_width=True)
                st.caption(
                    "Rojo: marginal de X₂ sin información adicional.  "
                    "Azul: distribución de X₂ dado que se observó X₁=x₁.  "
                    "La condicional es siempre más angosta (menor varianza) "
                    "y su centro se desplaza según ρ."
                )

        # VN3 — COMBINACIONES LINEALES
        with vn3:
            st.markdown(
                "**Teorema 4 — Combinaciones lineales de normales son normales.**  \n"
                "Si $(X_1, X_2) \\sim$ Normal bivariada y $Y = a_1 X_1 + a_2 X_2 + b$, entonces:"
            )
            st.latex(r"Y \sim N\!\left(b + a_1\mu_1 + a_2\mu_2,\;\; a_1^2\sigma_1^2 + a_2^2\sigma_2^2 + 2a_1 a_2\,\text{Cov}(X_1,X_2)\right)")
            st.markdown(
                "Aquí verificamos este teorema numéricamente: generamos muestras de $(X_1,X_2)$, "
                "calculamos $Y$, y verificamos que su distribución coincide con la predicción del Teorema 4."
            )
            col_vn3a, col_vn3b = st.columns([1,2])
            with col_vn3a:
                mu1_vn3 = st.slider("μ₁", -2.0, 2.0, 1.0, 0.5, key="vn3_m1")
                mu2_vn3 = st.slider("μ₂", -2.0, 2.0, -1.0, 0.5, key="vn3_m2")
                s1_vn3  = st.slider("σ₁", 0.3, 2.0, 1.0, 0.1, key="vn3_s1")
                s2_vn3  = st.slider("σ₂", 0.3, 2.0, 1.5, 0.1, key="vn3_s2")
                rho_vn3 = st.slider("ρ", -0.95, 0.95, 0.6, 0.05, key="vn3_rho")
                a1_vn3  = st.slider("Coeficiente a₁", -2.0, 2.0, 1.0, 0.1, key="vn3_a1")
                a2_vn3  = st.slider("Coeficiente a₂", -2.0, 2.0, 1.0, 0.1, key="vn3_a2")
                b_vn3   = st.slider("Constante b", -3.0, 3.0, 0.0, 0.5, key="vn3_b")

                cov_vn3 = rho_vn3*s1_vn3*s2_vn3
                EY = b_vn3 + a1_vn3*mu1_vn3 + a2_vn3*mu2_vn3
                VarY = a1_vn3**2*s1_vn3**2 + a2_vn3**2*s2_vn3**2 + 2*a1_vn3*a2_vn3*cov_vn3
                VarY = max(VarY, 1e-9)
                sigY = np.sqrt(VarY)

                rng_vn3 = np.random.default_rng(55)
                smp_vn3 = rng_vn3.multivariate_normal(
                    [mu1_vn3, mu2_vn3],
                    [[s1_vn3**2, cov_vn3],[cov_vn3, s2_vn3**2]],
                    3000
                )
                Y_sim = b_vn3 + a1_vn3*smp_vn3[:,0] + a2_vn3*smp_vn3[:,1]

                st.markdown("---")
                st.markdown(f"**E[Y] (Teorema 4):** `{EY:.3f}`")
                st.markdown(f"**E[Y] (simulación):** `{float(np.mean(Y_sim)):.3f}`")
                st.markdown(f"**σ(Y) (Teorema 4):** `{sigY:.3f}`")
                st.markdown(f"**σ(Y) (simulación):** `{float(np.std(Y_sim)):.3f}`")
                st.success(f"Y ~ N({EY:.2f}, {VarY:.3f})")

            with col_vn3b:
                y_grid = np.linspace(Y_sim.min(), Y_sim.max(), 300)
                f_teo  = stats.norm(EY, sigY).pdf(y_grid)

                fig_vn3 = go.Figure()
                fig_vn3.add_trace(go.Histogram(
                    x=Y_sim, histnorm='probability density',
                    name='Simulación de Y',
                    marker_color='#2196F3', opacity=0.6,
                    nbinsx=40
                ))
                fig_vn3.add_trace(go.Scatter(
                    x=y_grid, y=f_teo, mode='lines',
                    name=f'N({EY:.2f}, {VarY:.3f}) — Teorema 4',
                    line=dict(color='#e63946', width=3)
                ))
                fig_vn3.update_layout(
                    xaxis_title="y", yaxis_title="Densidad",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white', height=400,
                    margin=dict(l=0, r=0, t=20, b=0),
                    legend=dict(font=dict(color='white')),
                    barmode='overlay'
                )
                fig_vn3.update_xaxes(gridcolor='#222')
                fig_vn3.update_yaxes(gridcolor='#222')
                st.plotly_chart(fig_vn3, use_container_width=True)
                st.caption(
                    "Azul: histograma de 3000 simulaciones de Y = b + a₁X₁ + a₂X₂.  "
                    "Rojo: curva normal N(E[Y], Var(Y)) con parámetros del Teorema 4.  "
                    "La coincidencia confirma que la combinación lineal de normales es normal."
                )


# ═══════════════════════════════════════════════════════════════════════════
# PESTAÑA 6 — EJERCICIOS RESUELTOS
# ═══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown("## Ejercicios Resueltos")
    st.markdown(
        "Esta sección presenta un ejercicio representativo por cada tema del curso, "
        "resuelto paso a paso con la metodología vista en clase. "
        "El objetivo es que puedas identificar el **patrón de resolución** "
        "y aplicarlo a problemas similares."
    )

    ej1, ej2, ej3, ej4, ej5 = st.tabs([
        "📐 Ej. 1 — Independencia de V.A.",
        "🔗 Ej. 2 — Independencia de Vectores",
        "📊 Ej. 3 — Covarianza",
        "📈 Ej. 4 — Correlación",
        "🔔 Ej. 5 — Normal Multivariada"
    ])

    # ══════════════════════════════════════════════════════════════════════
    # EJERCICIO 1 — INDEPENDENCIA DE V.A. (EJERCICIO DE LA IMAGEN)
    # ══════════════════════════════════════════════════════════════════════
    with ej1:
        st.markdown("### Ejercicio — Independencia en un supermercado")
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Enunciado</span>', unsafe_allow_html=True)
        st.markdown(
            "Un supermercado local tiene **tres mostradores** de pago. "
            "Dos clientes llegan a los mostradores en momentos diferentes "
            "cuando los mostradores no atienden a otros clientes. "
            "Cada cliente elige un mostrador **al azar**, independientemente del otro.  \n\n"
            "Sea $X_1$ el número de clientes que eligen el mostrador 1 "
            "y $X_2$ el número de clientes que eligen el mostrador 2."
        )
        st.markdown(
            "**a)** ¿$X_1$ y $X_2$ son independientes? Si no son independientes, "
            "¿la dependencia es completa o parcial? Justifique.  \n"
            "**b)** Calcule $E[X_2 \\mid X_1]$ y $E[X_2]$. "
            "¿Qué relación existe entre ambos? ¿Es coherente con el ítem anterior?"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Resolución")

        # ── PARTE A ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Parte a) — ¿Son X₁ y X₂ independientes?</div>', unsafe_allow_html=True)

        st.markdown(
            "**Paso 1 — Definir el espacio muestral.**  \n"
            "Cada cliente elige uno de tres mostradores $\\{1, 2, 3\\}$ con igual probabilidad $\\frac{1}{3}$. "
            "Como los clientes eligen independientemente, el espacio muestral de la elección conjunta es:"
        )
        st.latex(r"\Omega = \{1,2,3\}^2 = \{(i,j) : i,j \in \{1,2,3\}\}, \quad |\Omega| = 9")
        st.markdown(
            "Cada par $(i,j)$ tiene probabilidad $\\frac{1}{9}$.  \n"
            "Aquí $i$ = mostrador elegido por el cliente 1, $j$ = mostrador elegido por el cliente 2."
        )

        st.markdown(
            "**Paso 2 — Identificar los valores posibles de $X_1$ y $X_2$.**  \n"
            "$X_1 = $ número de clientes que eligen el mostrador 1: puede ser $0, 1$ o $2$.  \n"
            "$X_2 = $ número de clientes que eligen el mostrador 2: puede ser $0, 1$ o $2$.  \n\n"
            "Nótese que $X_1 + X_2 \\leq 2$ (no pueden ir más de 2 clientes en total a los mostradores 1 y 2)."
        )

        st.markdown(
            "**Paso 3 — Construir la distribución conjunta $p_{1,2}(x_1, x_2)$.**  \n"
            "Contamos los pares $(i,j) \\in \\Omega$ que corresponden a cada combinación $(x_1, x_2)$:"
        )

        # Tabla interactiva con explicación
        col_t1, col_t2 = st.columns([1, 1])
        with col_t1:
            st.markdown("**Tabla conjunta $p_{1,2}(x_1, x_2)$:**")
            import pandas as pd
            tabla_joint = pd.DataFrame(
                {
                    "$x_2 = 0$": ["4/9", "2/9", "0"],
                    "$x_2 = 1$": ["2/9", "1/9", "0"],
                    "$x_2 = 2$": ["0",   "0",   "0"],
                },
                index=["$x_1=0$", "$x_1=1$", "$x_1=2$"]
            )
            st.table(tabla_joint)
            st.markdown(
                "*Verificación:* $4/9+2/9+2/9+1/9 = 9/9 = 1$ ✓  \n"
                "Nota: $(x_1,x_2)=(2,0),(0,2),(1,2),(2,1)$ también aparecen — "
                "pero $x_1+x_2 \\leq 2$ elimina algunos casos."
            )

        with col_t2:
            st.markdown("**¿Cómo se obtiene cada celda?**")
            st.markdown(
                "$(x_1=0, x_2=0)$: ningún cliente va al 1 ni al 2. "
                "Ambos van al 3: solo $(3,3)$ → $1/9$.  \n\n"
                "Espera — recalculemos sistemáticamente."
            )

        # Resolución completa de la tabla
        with st.expander("📐 Construcción detallada de la tabla conjunta"):
            st.markdown("Listamos todos los 9 pares y los clasificamos:")
            datos = {
                "Par (i,j)": ["(1,1)","(1,2)","(1,3)","(2,1)","(2,2)","(2,3)","(3,1)","(3,2)","(3,3)"],
                "X₁": [2,1,1,1,0,0,1,0,0],
                "X₂": [0,1,0,1,2,1,0,1,0],
                "P": ["1/9"]*9
            }
            df_pares = pd.DataFrame(datos)
            st.dataframe(df_pares, use_container_width=True)
            st.markdown(
                "Agrupando por $(x_1, x_2)$:  \n"
                "- $(2,0)$: solo $(1,1)$ → $p=1/9$  \n"
                "- $(1,1)$: $(1,2)$ y $(3,1)$ → $p=2/9$  \n"
                "- $(1,0)$: $(1,3)$ y $(3,1)$... — revisemos con cuidado:"
            )
            st.markdown(
                "- $(x_1=2, x_2=0)$: $(1,1)$ → $1/9$  \n"
                "- $(x_1=1, x_2=1)$: $(1,2)$ y $(2,1)$ → $2/9$  \n"
                "- $(x_1=1, x_2=0)$: $(1,3)$ y $(3,1)$ → $2/9$  \n"
                "- $(x_1=0, x_2=2)$: $(2,2)$ → $1/9$  \n"
                "- $(x_1=0, x_2=1)$: $(2,3)$ y $(3,2)$ → $2/9$  \n"
                "- $(x_1=0, x_2=0)$: $(3,3)$ → $1/9$  \n"
            )

        # Tabla correcta completa
        st.markdown("**Tabla conjunta correcta:**")
        st.latex(r"""
        \begin{array}{c|ccc|c}
        x_2 \backslash x_1 & 0 & 1 & 2 & p_2(x_2) \\ \hline
        0 & 1/9 & 2/9 & 1/9 & 4/9 \\
        1 & 2/9 & 2/9 & 0   & 4/9 \\
        2 & 1/9 & 0   & 0   & 1/9 \\ \hline
        p_1(x_1) & 4/9 & 4/9 & 1/9 & 1
        \end{array}
        """)

        st.markdown(
            "**Paso 4 — Calcular las marginales.**  \n"
            "Marginal de $X_1$ (suma por filas):"
        )
        st.latex(r"p_1(0) = \frac{4}{9}, \quad p_1(1) = \frac{4}{9}, \quad p_1(2) = \frac{1}{9}")
        st.markdown("Marginal de $X_2$ (suma por columnas):")
        st.latex(r"p_2(0) = \frac{4}{9}, \quad p_2(1) = \frac{4}{9}, \quad p_2(2) = \frac{1}{9}")
        st.markdown(
            "> 💡 Por simetría del problema ($X_1$ y $X_2$ tienen el mismo rol), "
            "es esperable que tengan la misma distribución marginal."
        )

        st.markdown(
            "**Paso 5 — Verificar el criterio de factorización.**  \n"
            "Para que $X_1 \\perp X_2$, debe cumplirse $p_{1,2}(x_1,x_2) = p_1(x_1)\\cdot p_2(x_2)$ "
            "para **todos** los valores.  \n\n"
            "Revisamos una celda:"
        )
        st.latex(r"p_1(1)\cdot p_2(1) = \frac{4}{9}\cdot\frac{4}{9} = \frac{16}{81}")
        st.latex(r"p_{1,2}(1,1) = \frac{2}{9} = \frac{18}{81}")
        st.latex(r"\frac{16}{81} \neq \frac{18}{81}")

        st.markdown('<div class="obs-box"><span class="label-obs">Conclusión parte a)</span>', unsafe_allow_html=True)
        st.markdown(
            "Como $p_{1,2}(1,1) \\neq p_1(1)\\cdot p_2(1)$, **el criterio de factorización no se cumple**.  \n"
            "$X_1$ y $X_2$ **no son independientes**.  \n\n"
            "**¿La dependencia es completa o parcial?**  \n"
            "Buscamos si existe $g$ tal que $X_2 = g(X_1)$.  \n"
            "Cuando $X_1 = 0$: $X_2$ puede valer $0$, $1$ o $2$ (no hay un único valor).  \n"
            "Por tanto, $X_1$ **no determina completamente** a $X_2$.  \n"
            "La dependencia es **parcial**."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            "> 💡 **Intuición:** si el cliente 1 eligió el mostrador 1 ($X_1 = 1$), "
            "el cliente 2 ya no puede elegir también el 1 ni el 2 con total libertad — "
            "saber $X_1$ sí restringe la distribución de $X_2$, pero no la fija completamente."
        )

        # ── PARTE B ────────────────────────────────────────────────────────
        st.markdown('<div class="section-title">Parte b) — E[X₂ | X₁] y E[X₂]</div>', unsafe_allow_html=True)

        st.markdown(
            "**Paso 1 — Calcular $E[X_2 \\mid X_1 = x_1]$ para cada valor de $x_1$.**  \n"
            "La esperanza condicional requiere primero la distribución condicional "
            "$p_{2|1}(x_2 \\mid x_1) = p_{1,2}(x_1,x_2)/p_1(x_1)$."
        )

        st.markdown("**Caso $X_1 = 0$** ($p_1(0) = 4/9$):")
        st.latex(r"p_{2|1}(0|0)=\frac{1/9}{4/9}=\frac{1}{4}, \quad p_{2|1}(1|0)=\frac{2/9}{4/9}=\frac{2}{4}, \quad p_{2|1}(2|0)=\frac{1/9}{4/9}=\frac{1}{4}")
        st.latex(r"E[X_2\mid X_1=0] = 0\cdot\frac{1}{4}+1\cdot\frac{2}{4}+2\cdot\frac{1}{4} = \frac{0+2+2}{4} = 1")

        st.markdown("**Caso $X_1 = 1$** ($p_1(1) = 4/9$):")
        st.latex(r"p_{2|1}(0|1)=\frac{2/9}{4/9}=\frac{1}{2}, \quad p_{2|1}(1|1)=\frac{2/9}{4/9}=\frac{1}{2}, \quad p_{2|1}(2|1)=0")
        st.latex(r"E[X_2\mid X_1=1] = 0\cdot\frac{1}{2}+1\cdot\frac{1}{2} = \frac{1}{2}")

        st.markdown("**Caso $X_1 = 2$** ($p_1(2) = 1/9$):")
        st.latex(r"p_{2|1}(0|2)=\frac{1/9}{1/9}=1, \quad p_{2|1}(1|2)=0, \quad p_{2|1}(2|2)=0")
        st.latex(r"E[X_2\mid X_1=2] = 0\cdot 1 = 0")

        st.markdown("**Por tanto, la esperanza condicional $E[X_2 \\mid X_1]$ es una variable aleatoria:**")
        st.latex(r"E[X_2\mid X_1] = \begin{cases} 1 & \text{si } X_1=0 \\ 1/2 & \text{si } X_1=1 \\ 0 & \text{si } X_1=2 \end{cases}")

        st.markdown("**Paso 2 — Calcular $E[X_2]$ directamente (por la marginal):**")
        st.latex(r"E[X_2] = 0\cdot\frac{4}{9}+1\cdot\frac{4}{9}+2\cdot\frac{1}{9} = 0+\frac{4}{9}+\frac{2}{9} = \frac{6}{9} = \frac{2}{3}")

        st.markdown("**Paso 3 — Verificar la Ley de la Esperanza Total:**")
        st.latex(r"E\bigl[E[X_2\mid X_1]\bigr] = 1\cdot\frac{4}{9}+\frac{1}{2}\cdot\frac{4}{9}+0\cdot\frac{1}{9} = \frac{4}{9}+\frac{2}{9} = \frac{6}{9} = \frac{2}{3} = E[X_2] \;\checkmark")

        st.markdown('<div class="obs-box"><span class="label-obs">Conclusión parte b)</span>', unsafe_allow_html=True)
        st.markdown(
            "$E[X_2 \\mid X_1]$ **depende del valor de $X_1$** — cambia según lo que se observe.  \n"
            "$E[X_2] = 2/3$ es una constante.  \n\n"
            "**Relación:** $E[X_2 \\mid X_1] \\neq E[X_2]$ para todos los valores de $X_1$.  \n"
            "Esto es **coherente** con el ítem a): si $X_1 \\perp X_2$, "
            "se tendría $E[X_2 \\mid X_1 = x_1] = E[X_2]$ para todo $x_1$. "
            "El hecho de que no sean iguales confirma la dependencia."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Visualización del ejercicio
        st.markdown("---")
        st.markdown("#### Visualización — Distribuciones condicionales vs. marginal")
        x2_vals = [0, 1, 2]
        cond_x1_0 = [1/4, 2/4, 1/4]
        cond_x1_1 = [1/2, 1/2, 0]
        cond_x1_2 = [1, 0, 0]
        marg_x2   = [4/9, 4/9, 1/9]

        fig_ej1 = go.Figure()
        for y_vals, name, color in [
            (cond_x1_0, "f(X₂|X₁=0)", "#2196F3"),
            (cond_x1_1, "f(X₂|X₁=1)", "#4CAF50"),
            (cond_x1_2, "f(X₂|X₁=2)", "#FF9800"),
            (marg_x2,   "Marginal f₂(x₂)", "#e63946"),
        ]:
            fig_ej1.add_trace(go.Bar(
                x=[str(v) for v in x2_vals],
                y=y_vals, name=name,
                marker_color=color, opacity=0.8
            ))
        fig_ej1.update_layout(
            barmode='group',
            xaxis_title="x₂", yaxis_title="Probabilidad",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(color='white'))
        )
        fig_ej1.update_xaxes(gridcolor='#222')
        fig_ej1.update_yaxes(gridcolor='#222')
        st.plotly_chart(fig_ej1, use_container_width=True)
        st.caption(
            "Cada color muestra cómo cambia la distribución de X₂ según el valor observado de X₁.  "
            "Si fueran independientes, las tres distribuciones condicionales (azul, verde, naranja) "
            "serían idénticas a la marginal (rojo). "
            "Como no lo son, X₁ y X₂ son dependientes."
        )

    # ══════════════════════════════════════════════════════════════════════
    # EJERCICIO 2 — INDEPENDENCIA DE VECTORES
    # ══════════════════════════════════════════════════════════════════════
    with ej2:
        st.markdown("### Ejercicio — Independencia en clasificación de emisores corporativos")
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Enunciado</span>', unsafe_allow_html=True)
        st.markdown(
            "Una agencia de clasificación de riesgo evalúa emisores de deuda corporativa según dos variables:  \n"
            "- $X_1$: si el emisor honró sus obligaciones en el último período ($X_1=1$: sin incumplimiento) o no ($X_1=0$: incumplimiento).  \n"
            "- $X_2$: calidad del gobierno corporativo: deficiente ($X_2=0$), aceptable ($X_2=1$) o sólido ($X_2=2$).  \n\n"
            "Se define $Y = X_1 + X_2$ como un **índice de salud financiera** del emisor.  \n\n"
            "La función de probabilidad conjunta de $(X_1, X_2)$ es:"
        )
        st.latex(r"""
        \begin{array}{c|cc|c}
        x_2 \backslash x_1 & 0 & 1 & p_2(x_2) \\ \hline
        0 & 0.20 & 0.10 & 0.30 \\
        1 & 0.15 & 0.15 & 0.30 \\
        2 & 0.05 & 0.35 & 0.40 \\ \hline
        p_1(x_1) & 0.40 & 0.60 & 1.00
        \end{array}
        """)
        st.markdown(
            "**a)** ¿Son $X_1$ y $X_2$ independientes? Justifique con el criterio de factorización.  \n"
            "**b)** ¿$X_1$ determina completamente a $X_2$? ¿Y $X_2$ a $X_1$? Discuta.  \n"
            "**c)** Calcule $E[Y \\mid X_1 = 1]$, es decir, el índice de salud financiera esperado dado que el emisor no incurrió en incumplimiento.  \n"
            "**d)** ¿Es $(X_1, X_2)$ como vector independiente de $Y$? Discuta el concepto."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Resolución")

        st.markdown('<div class="section-title">Parte a) — Criterio de factorización</div>', unsafe_allow_html=True)
        st.markdown(
            "El criterio establece que $X_1 \\perp X_2$ si y solo si "
            "$p_{1,2}(x_1,x_2) = p_1(x_1)\\cdot p_2(x_2)$ para **todos** los pares $(x_1,x_2)$.  \n\n"
            "Las marginales ya están en la tabla: $p_1(0)=0.40$, $p_1(1)=0.60$, "
            "$p_2(0)=0.30$, $p_2(1)=0.30$, $p_2(2)=0.40$.  \n\n"
            "Verificamos **todas** las celdas:"
        )
        datos_fact = {
            "Par (x1, x2)": ["(0,0)","(0,1)","(0,2)","(1,0)","(1,1)","(1,2)"],
            "p(x1,x2)": [0.20, 0.15, 0.05, 0.10, 0.15, 0.35],
            "p1(x1)": [0.40, 0.40, 0.40, 0.60, 0.60, 0.60],
            "p2(x2)": [0.30, 0.30, 0.40, 0.30, 0.30, 0.40],
            "p1 × p2": [0.12, 0.12, 0.16, 0.18, 0.18, 0.24],
            "¿Iguales?": ["❌","❌","❌","❌","❌","❌"]
        }
        st.dataframe(pd.DataFrame(datos_fact), use_container_width=True)
        st.markdown(
            "Basta con encontrar **una** celda donde no se cumpla para concluir dependencia. "
            "En este caso, **ninguna** celda cumple la factorización: "
            "$p_{1,2}(x_1,x_2) \\neq p_1(x_1)\\cdot p_2(x_2)$ en todos los pares."
        )
        st.markdown('<div class="obs-box"><span class="label-obs">Conclusión</span>', unsafe_allow_html=True)
        st.markdown(
            "$X_1$ y $X_2$ **no son independientes**: la tabla no factoriza.  \n"
            "**Interpretación financiera:** si un emisor honró sus obligaciones ($X_1=1$), "
            "la probabilidad de tener gobierno corporativo sólido ($X_2=2$) cambia respecto a la marginal — "
            "el historial de pago sí informa sobre la calidad de gestión del emisor."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte b) — Determinación completa o parcial</div>', unsafe_allow_html=True)
        st.markdown(
            "**¿$X_1$ determina completamente a $X_2$?**  \n"
            "Cuando $X_1=0$: $X_2$ puede ser $0$, $1$ o $2$ (los tres son posibles).  \n"
            "No existe una función $g$ tal que $X_2 = g(X_1)$ — $X_1$ **no determina completamente** a $X_2$.  \n\n"
            "**¿$X_2$ determina completamente a $X_1$?**  \n"
            "Cuando $X_2=0$: $X_1$ puede ser $0$ o $1$. Tampoco hay función $h$ tal que $X_1=h(X_2)$.  \n\n"
            "**Conclusión:** la dependencia es **parcial** en ambas direcciones. "
            "Ninguna variable fija completamente a la otra, pero el conocimiento de una "
            "sí modifica la distribución de la otra."
        )

        st.markdown('<div class="section-title">Parte c) — E[Y | X₁ = 1]</div>', unsafe_allow_html=True)
        st.markdown(
            "Recordemos que $Y = X_1 + X_2$. Dado $X_1=1$, tenemos $Y = 1 + X_2$.  \n"
            "Por linealidad de la esperanza condicional:"
        )
        st.latex(r"E[Y \mid X_1=1] = E[X_1 + X_2 \mid X_1=1] = 1 + E[X_2 \mid X_1=1]")
        st.markdown("Calculamos $E[X_2 \\mid X_1=1]$ usando la distribución condicional:")
        st.latex(r"p_{2|1}(x_2|X_1=1): \quad p(0|1)=\frac{0.10}{0.60}=\frac{1}{6},\; p(1|1)=\frac{0.15}{0.60}=\frac{1}{4},\; p(2|1)=\frac{0.35}{0.60}=\frac{7}{12}")
        st.latex(r"E[X_2\mid X_1=1] = 0\cdot\frac{1}{6}+1\cdot\frac{1}{4}+2\cdot\frac{7}{12} = 0+\frac{1}{4}+\frac{14}{12} = \frac{3}{12}+\frac{14}{12} = \frac{17}{12}")
        st.latex(r"E[Y\mid X_1=1] = 1+\frac{17}{12} = \frac{29}{12} \approx 2.42")

        st.markdown('<div class="obs-box"><span class="label-obs">Interpretación</span>', unsafe_allow_html=True)
        st.markdown(
            "Cuando el emisor no incumple ($X_1=1$), el índice de salud financiera esperado es $\\approx 2.42$.  \n"
            "Para comparar, calculemos $E[Y]$ sin información:  \n"
            "$E[X_1] = 0.60$, $E[X_2] = 0\\cdot0.30+1\\cdot0.30+2\\cdot0.40 = 1.10$.  \n"
            "$E[Y] = E[X_1]+E[X_2] = 0.60+1.10 = 1.70$.  \n\n"
            "Saber que el emisor no incumplió eleva el índice esperado de $1.70$ a $2.42$ — "
            "esto confirma la dependencia: el historial de pago **aporta información** sobre el índice de salud."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte d) — Independencia del vector (X₁, X₂) e Y</div>', unsafe_allow_html=True)
        st.markdown(
            "El vector $(X_1,X_2)$ y la variable $Y = X_1+X_2$ son **siempre dependientes** "
            "cuando $Y$ es función no constante de $(X_1,X_2)$.  \n\n"
            "Esto es porque el vector $(X_1,X_2)$ **determina completamente** a $Y$: "
            "dado cualquier par $(x_1,x_2)$, el valor $y = x_1+x_2$ queda fijado.  \n\n"
            "Por tanto, el evento $\\{Y=3\\} = \\{(X_1,X_2)=(1,2)\\}$ está "
            "**generado por el vector** — no puede ser independiente de él.  \n\n"
            "La independencia de vectores cobra sentido cuando los bloques son **conceptualmente separados**: "
            "por ejemplo, el vector de características de un emisor vs. el vector de características "
            "de otro emisor de una industria completamente distinta — en ese caso sí podría argumentarse "
            "independencia entre los dos vectores."
        )

        # Visualización
        st.markdown("---")
        st.markdown("#### Visualización — Distribución de la calidad de gobierno corporativo según historial de pago")
        p_x2_dado_x1_0 = [0.20/0.40, 0.15/0.40, 0.05/0.40]
        p_x2_dado_x1_1 = [0.10/0.60, 0.15/0.60, 0.35/0.60]
        p_x2_marg      = [0.30, 0.30, 0.40]

        fig_ej2 = go.Figure()
        for vals, name, col in [
            (p_x2_marg,      "Marginal X₂ (sin info)",        "#e63946"),
            (p_x2_dado_x1_0, "X₂ | X₁=0 (incumplimiento)",   "#FF9800"),
            (p_x2_dado_x1_1, "X₂ | X₁=1 (sin incumplimiento)","#4CAF50"),
        ]:
            fig_ej2.add_trace(go.Bar(
                x=["Deficiente (X₂=0)", "Aceptable (X₂=1)", "Sólido (X₂=2)"],
                y=vals, name=name,
                marker_color=col, opacity=0.85
            ))
        fig_ej2.update_layout(
            barmode='group', yaxis_title="Probabilidad",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white', height=380,
            margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(font=dict(color='white'))
        )
        fig_ej2.update_xaxes(gridcolor='#222')
        fig_ej2.update_yaxes(gridcolor='#222', range=[0,0.7])
        st.plotly_chart(fig_ej2, use_container_width=True)
        st.caption(
            "Rojo: distribución marginal de la calidad de gobierno corporativo sin información adicional.  "
            "Naranja: dado que el emisor incurrió en incumplimiento.  "
            "Verde: dado que el emisor honró sus obligaciones.  "
            "Si fueran independientes, las tres distribuciones serían idénticas."
        )

    # ══════════════════════════════════════════════════════════════════════
    # EJERCICIO 3 — COVARIANZA (CONTINUO)
    # ══════════════════════════════════════════════════════════════════════
    with ej3:
        st.markdown("### Ejercicio — Covarianza en un modelo de scoring crediticio")
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Enunciado</span>', unsafe_allow_html=True)
        st.markdown(
            "En un modelo de scoring crediticio, se modelan conjuntamente dos variables continuas de un solicitante: "
            "$X$: índice de capacidad de pago normalizado ($0 < x < 1$, donde valores cercanos a 1 "
            "indican mayor capacidad) e $Y$: historial de comportamiento crediticio normalizado "
            "($0 < y < 2$, donde valores altos indican buen historial), "
            "con la siguiente densidad conjunta:"
        )
        st.latex(r"f(x,y) = \begin{cases} c\,x\,y & 0 < x < 1,\; 0 < y < 2 \\ 0 & \text{en otro caso} \end{cases}")
        st.markdown(
            "Se puede verificar que $c=1$, por lo que la densidad conjunta es "
            "$f(x,y) = xy$, con $0<x<1$ y $0<y<2$.  \\n\\n"
            "**a)** Halle las densidades marginales $f_X(x)$ y $f_Y(y)$.  \\n"
            "**b)** ¿Son $X$ e $Y$ independientes? Justifique.  \\n"
            "**c)** Calcule $\\text{Cov}(X,Y)$ usando la fórmula operacional e interprete "
            "en el contexto del scoring.  \\n"
            "**d)** Calcule $\\text{Var}(3X - Y)$, donde $3X-Y$ puede interpretarse como "
            "un índice de riesgo ajustado."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Resolución")

        st.markdown('<div class="section-title">Parte a) — Densidades marginales</div>', unsafe_allow_html=True)
        st.markdown(
            "La marginal de $X$ se obtiene integrando $f(x,y)$ sobre todos los valores de $y$:"
        )
        st.latex(r"f_X(x) = \int_0^2 xy\;dy = x\cdot\left[\frac{y^2}{2}\right]_0^2 = x\cdot 2 = 2x, \quad 0<x<1")
        st.markdown("La marginal de $Y$:")
        st.latex(r"f_Y(y) = \int_0^1 xy\;dx = y\cdot\left[\frac{x^2}{2}\right]_0^1 = \frac{y}{2}, \quad 0<y<2")
        st.markdown(
            "> Verificación: $\\int_0^1 2x\,dx = 1$ ✓ y $\\int_0^2 y/2\,dy = 1$ ✓."
        )

        st.markdown('<div class="section-title">Parte b) — ¿Son independientes?</div>', unsafe_allow_html=True)
        st.markdown(
            "Comprobamos si $f(x,y) = f_X(x)\\cdot f_Y(y)$:"
        )
        st.latex(r"f_X(x)\cdot f_Y(y) = 2x\cdot\frac{y}{2} = xy = f(x,y) \quad \checkmark")
        st.markdown('<div class="prop-box"><span class="label-prop">Conclusión</span>', unsafe_allow_html=True)
        st.markdown(
            "**$X$ e $Y$ son independientes.** La densidad conjunta factoriza exactamente "
            "en el producto de las marginales.  \n\n"
            "Esto nos anticipará el resultado de la parte d): si son independientes, "
            "la covarianza debe ser cero."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte c) — Cov(X, Y)</div>', unsafe_allow_html=True)
        st.markdown(
            "Usamos la **fórmula operacional**: $\\text{Cov}(X,Y) = E[XY] - E[X]\\cdot E[Y]$.  \n\n"
            "**Calculamos $E[X]$:**"
        )
        st.latex(r"E[X] = \int_0^1 x\cdot 2x\;dx = 2\int_0^1 x^2\;dx = 2\cdot\frac{1}{3} = \frac{2}{3}")
        st.markdown("**Calculamos $E[Y]$:**")
        st.latex(r"E[Y] = \int_0^2 y\cdot\frac{y}{2}\;dy = \frac{1}{2}\int_0^2 y^2\;dy = \frac{1}{2}\cdot\frac{8}{3} = \frac{4}{3}")
        st.markdown("**Calculamos $E[XY]$:**")
        st.latex(r"E[XY] = \int_0^1\int_0^2 xy\cdot xy\;dy\,dx = \int_0^1\int_0^2 x^2 y^2\;dy\,dx")
        st.latex(r"= \int_0^1 x^2\;dx\cdot\int_0^2 y^2\;dy = \frac{1}{3}\cdot\frac{8}{3} = \frac{8}{9}")
        st.markdown("**Aplicamos la fórmula:**")
        st.latex(r"\text{Cov}(X,Y) = E[XY] - E[X]\cdot E[Y] = \frac{8}{9} - \frac{2}{3}\cdot\frac{4}{3} = \frac{8}{9} - \frac{8}{9} = 0")

        st.markdown('<div class="obs-box"><span class="label-obs">Interpretación</span>', unsafe_allow_html=True)
        st.markdown(
            "$\\text{Cov}(X,Y) = 0$: el resultado es **consistente** con la parte c).  \n\n"
            "Recordemos la Propiedad 3: si $X \\perp Y$ entonces $\\text{Cov}(X,Y)=0$.  \n"
            "Aquí comprobamos esa propiedad directamente con la fórmula operacional.  \n\n"
            "**En el contexto del problema:** la tasa de ocupación $X$ y el ingreso relativo $Y$ "
            "no tienen ninguna relación lineal — en promedio, conocer la ocupación no "
            "predice el ingreso, y viceversa."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte d) — Var(3X - Y)</div>', unsafe_allow_html=True)
        st.markdown("Aplicamos la Propiedad 5 de la covarianza:")
        st.latex(r"\text{Var}(3X-Y) = 9\,\text{Var}(X) + \text{Var}(Y) - 2\cdot 3\cdot 1\cdot\text{Cov}(X,Y)")
        st.markdown(
            "Como $\\text{Cov}(X,Y)=0$, el término cruzado desaparece:  \n"
            "$\\text{Var}(3X-Y) = 9\\,\\text{Var}(X) + \\text{Var}(Y)$"
        )
        st.markdown("**Calculamos $\\text{Var}(X)$:**")
        st.latex(r"E[X^2] = \int_0^1 x^2\cdot 2x\;dx = 2\int_0^1 x^3\;dx = 2\cdot\frac{1}{4} = \frac{1}{2}")
        st.latex(r"\text{Var}(X) = E[X^2]-(E[X])^2 = \frac{1}{2}-\left(\frac{2}{3}\right)^2 = \frac{1}{2}-\frac{4}{9} = \frac{9-8}{18} = \frac{1}{18}")
        st.markdown("**Calculamos $\\text{Var}(Y)$:**")
        st.latex(r"E[Y^2] = \int_0^2 y^2\cdot\frac{y}{2}\;dy = \frac{1}{2}\int_0^2 y^3\;dy = \frac{1}{2}\cdot 4 = 2")
        st.latex(r"\text{Var}(Y) = E[Y^2]-(E[Y])^2 = 2-\left(\frac{4}{3}\right)^2 = 2-\frac{16}{9} = \frac{18-16}{9} = \frac{2}{9}")
        st.markdown("**Resultado final:**")
        st.latex(r"\text{Var}(3X-Y) = 9\cdot\frac{1}{18}+\frac{2}{9} = \frac{1}{2}+\frac{2}{9} = \frac{9+4}{18} = \frac{13}{18}")
        st.markdown('<div class="prop-box"><span class="label-prop">Resultado</span>', unsafe_allow_html=True)
        st.latex(r"\text{Var}(3X-Y) = \frac{13}{18} \approx 0.722")
        st.markdown(
            "> **Interpretación:** la varianza del índice de riesgo ajustado $3X-Y$ "
            "se construye únicamente con las varianzas individuales, sin término cruzado, "
            "porque la capacidad de pago y el historial son independientes. "
            "Si hubiera dependencia, el término $-6\\,\\text{Cov}(X,Y)$ modificaría el riesgo total."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Visualización de las marginales
        st.markdown("---")
        st.markdown("#### Visualización — Densidades marginales y densidad conjunta")
        col_ej3a, col_ej3b = st.columns(2)
        with col_ej3a:
            x_plot = np.linspace(0, 1, 200)
            y_plot = np.linspace(0, 2, 200)
            fig_mx = go.Figure()
            fig_mx.add_trace(go.Scatter(
                x=x_plot, y=2*x_plot, mode='lines',
                line=dict(color='#2196F3', width=3),
                fill='tozeroy', fillcolor='rgba(33,150,243,0.2)',
                name='f_X(x) = 2x'
            ))
            fig_mx.update_layout(
                title="Marginal de X (capacidad de pago)", xaxis_title="x", yaxis_title="f_X(x)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=280, margin=dict(l=0,r=0,t=40,b=0)
            )
            fig_mx.update_xaxes(gridcolor='#222')
            fig_mx.update_yaxes(gridcolor='#222')
            st.plotly_chart(fig_mx, use_container_width=True)

        with col_ej3b:
            fig_my = go.Figure()
            fig_my.add_trace(go.Scatter(
                x=y_plot, y=y_plot/2, mode='lines',
                line=dict(color='#4CAF50', width=3),
                fill='tozeroy', fillcolor='rgba(76,175,80,0.2)',
                name='f_Y(y) = y/2'
            ))
            fig_my.update_layout(
                title="Marginal de Y (historial crediticio)", xaxis_title="y", yaxis_title="f_Y(y)",
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=280, margin=dict(l=0,r=0,t=40,b=0)
            )
            fig_my.update_xaxes(gridcolor='#222')
            fig_my.update_yaxes(gridcolor='#222')
            st.plotly_chart(fig_my, use_container_width=True)

        st.caption(
            "Ambas densidades marginales son crecientes: solicitantes con mayor capacidad de pago "
            "e historial más sólido son más frecuentes en la base. "
            "Como f(x,y)=xy = f_X(x)·f_Y(y), las variables son independientes — "
            "en este modelo, la capacidad de pago no predice el historial crediticio."
        )

    # ══════════════════════════════════════════════════════════════════════
    # EJERCICIO 4 — CORRELACIÓN
    # ══════════════════════════════════════════════════════════════════════
    with ej4:
        st.markdown("### Ejercicio — Correlación entre retornos y su invarianza en trading")
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Enunciado</span>', unsafe_allow_html=True)
        st.markdown(
            "En una mesa de trading, un analista cuantitativo trabaja con los retornos diarios "
            "$X$ e $Y$ de dos activos financieros y observa que tienen correlación $\\rho_{X,Y}$. "
            "Para construir estrategias, reescala los retornos: $Z = a + bX$ y $W = c + dY$, "
            "donde $b$ y $d$ pueden representar factores de apalancamiento o cambios de unidad."
        )
        st.markdown(
            "**a)** Muestre que $|\\rho_{Z,W}| = |\\rho_{X,Y}|$.  \n"
            "**b)** ¿Bajo qué condición $\\rho_{Z,W} = \\rho_{X,Y}$ (sin valor absoluto)?  \n"
            "**c)** Aplique el resultado: los retornos diarios de una acción $X$ y un índice $Y$ "
            "tienen correlación $\\rho_{X,Y} = 0.65$. "
            "El trading desk reescala los retornos: $Z = 1.5\\,X$ "
            "(posición larga con apalancamiento 1.5) y $W = -Y$ "
            "(posición corta sobre el índice). ¿Cuál es la correlación entre $Z$ y $W$?"
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Resolución")

        st.markdown('<div class="section-title">Parte a) — Demostración: |ρ(Z,W)| = |ρ(X,Y)|</div>', unsafe_allow_html=True)
        st.markdown("**Paso 1 — Calcular Cov(Z,W).**  \nUsando la Propiedad 4 de covarianza:")
        st.latex(r"\text{Cov}(Z,W) = \text{Cov}(a+bX,\, c+dY) = bd\,\text{Cov}(X,Y)")
        st.markdown(
            "Las constantes aditivas $a$ y $c$ no afectan la covarianza "
            "(desplazan la media pero no la dispersión conjunta)."
        )

        st.markdown("**Paso 2 — Calcular $\\sigma_Z$ y $\\sigma_W$.**")
        st.latex(r"\text{Var}(Z) = \text{Var}(a+bX) = b^2\,\text{Var}(X) \;\Rightarrow\; \sigma_Z = |b|\,\sigma_X")
        st.latex(r"\text{Var}(W) = \text{Var}(c+dY) = d^2\,\text{Var}(Y) \;\Rightarrow\; \sigma_W = |d|\,\sigma_Y")

        st.markdown("**Paso 3 — Calcular $\\rho_{Z,W}$.**")
        st.latex(r"\rho_{Z,W} = \frac{\text{Cov}(Z,W)}{\sigma_Z\cdot\sigma_W} = \frac{bd\,\text{Cov}(X,Y)}{|b|\sigma_X\cdot|d|\sigma_Y}")
        st.latex(r"= \frac{bd}{|bd|}\cdot\frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} = \frac{bd}{|bd|}\cdot\rho_{X,Y}")

        st.markdown("**Paso 4 — Tomar valor absoluto.**")
        st.latex(r"|\rho_{Z,W}| = \left|\frac{bd}{|bd|}\right|\cdot|\rho_{X,Y}| = 1\cdot|\rho_{X,Y}| = |\rho_{X,Y}| \qquad \blacksquare")

        st.markdown('<div class="prop-box"><span class="label-prop">Conclusión de la demostración</span>', unsafe_allow_html=True)
        st.markdown(
            "El **valor absoluto** de la correlación es invariante bajo transformaciones lineales afines.  \n"
            "La correlación mide la **fuerza** de la relación lineal, que no cambia al reescalar o desplazar.  \n"
            "Solo el **signo** puede cambiar, y únicamente cuando $bd < 0$ (es decir, "
            "cuando una de las transformaciones invierte la dirección)."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte b) — ¿Cuándo ρ(Z,W) = ρ(X,Y) exactamente?</div>', unsafe_allow_html=True)
        st.markdown(
            "De la derivación: $\\rho_{Z,W} = \\frac{bd}{|bd|}\\cdot\\rho_{X,Y}$.  \n"
            "Para que $\\rho_{Z,W} = \\rho_{X,Y}$ (igualdad exacta, no solo en valor absoluto):  \n"
        )
        st.latex(r"\frac{bd}{|bd|} = 1 \;\Longleftrightarrow\; bd > 0")
        st.markdown(
            "Es decir, cuando $b$ y $d$ tienen el **mismo signo**: "
            "ambas transformaciones van en la misma dirección (ambas crecientes o ambas decrecientes).  \n"
            "Si $bd < 0$, la correlación cambia de signo: $\\rho_{Z,W} = -\\rho_{X,Y}$."
        )

        st.markdown('<div class="section-title">Parte c) — Aplicación: posición larga y corta en trading</div>', unsafe_allow_html=True)
        st.markdown(
            "Datos: $\\rho_{X,Y} = 0.65$, $Z = 1.5\\,X$ (luego $b=1.5>0$) "
            "y $W = -Y$ (luego $d=-1<0$).  \\n"
            "Calculamos el producto: $bd = (1.5)(-1) = -1.5 < 0$."
        )
        st.latex(r"\rho_{Z,W} = \frac{(1.5)(-1)}{|(1.5)(-1)|}\cdot\rho_{X,Y} = (-1)\cdot 0.65 = -0.65")

        st.markdown('<div class="obs-box"><span class="label-obs">Interpretación</span>', unsafe_allow_html=True)
        st.markdown(
            "Convertir de Celsius a Fahrenheit es una transformación lineal creciente ($b>0$): "
            "preserva el orden de los valores.  \n"
            "Por eso la correlación **no cambia**: la relación entre temperatura y presión arterial "
            "es la misma independientemente de la unidad que usemos para medir la temperatura.  \n\n"
            "Este resultado es fundamental en econometría: la correlación es **invariante a la escala** "
            "de medición, lo que la hace una medida robusta y comparable entre estudios."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Visualización interactiva
        st.markdown("---")
        st.markdown("#### Visualización — El signo de ρ cambia, pero |ρ| no")
        col_e4a, col_e4b = st.columns([1,2])
        with col_e4a:
            rho_base_e4 = st.slider("ρ(X,Y) base", -0.95, 0.95, 0.65, 0.05, key="e4_rho")
            b_e4 = st.slider("Coeficiente b (en Z=a+bX)", -3.0, 3.0, 1.8, 0.1, key="e4_b")
            d_e4 = st.slider("Coeficiente d (en W=c+dY)", -3.0, 3.0, 1.0, 0.1, key="e4_d")
            if b_e4 == 0 or d_e4 == 0:
                st.warning("Con b=0 o d=0, la variable se vuelve constante y ρ no está definida.")
            else:
                rho_zw = np.sign(b_e4*d_e4)*rho_base_e4
                st.markdown("---")
                bd_sign = "positivo (+1)" if b_e4*d_e4 > 0 else "negativo (−1)"
                st.markdown(f"**bd = {b_e4:.1f}×{d_e4:.1f} = {b_e4*d_e4:.2f} → {bd_sign}**")
                st.markdown(f"**ρ(Z,W):** `{rho_zw:.3f}`")
                st.markdown(f"**|ρ(Z,W)|:** `{abs(rho_zw):.3f}`")
                st.markdown(f"**|ρ(X,Y)|:** `{abs(rho_base_e4):.3f}`")
                if abs(abs(rho_zw)-abs(rho_base_e4)) < 0.001:
                    st.success("✅ |ρ(Z,W)| = |ρ(X,Y)| — demostrado.")
        with col_e4b:
            if b_e4 != 0 and d_e4 != 0:
                rng_e4 = np.random.default_rng(77)
                cov_e4 = rho_base_e4
                smp_e4 = rng_e4.multivariate_normal([0,0], [[1,cov_e4],[cov_e4,1]], 500)
                X_e4, Y_e4 = smp_e4[:,0], smp_e4[:,1]
                Z_e4 = 2 + b_e4*X_e4
                W_e4 = 1 + d_e4*Y_e4

                fig_e4 = make_subplots(rows=1, cols=2,
                    subplot_titles=[f"(X,Y)  ρ={rho_base_e4:.2f}",
                                    f"(Z,W)=(a+bX,c+dY)  ρ={rho_zw:.2f}"])
                for col_i, (xa, ya) in enumerate([(X_e4,Y_e4),(Z_e4,W_e4)], 1):
                    fig_e4.add_trace(go.Scatter(
                        x=xa, y=ya, mode='markers',
                        marker=dict(color='#2196F3', size=3, opacity=0.4),
                        showlegend=False
                    ), row=1, col=col_i)
                fig_e4.update_layout(
                    height=350,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font_color='white',
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                for ci in range(1,3):
                    fig_e4.update_xaxes(gridcolor='#222', row=1, col=ci)
                    fig_e4.update_yaxes(gridcolor='#222', row=1, col=ci)
                st.plotly_chart(fig_e4, use_container_width=True)
                st.caption(
                    "La nube izquierda (X,Y) y la derecha (Z,W) pueden diferir en escala e inclinación, "
                    "pero la fuerza de la relación lineal — medida por |ρ| — es la misma. "
                    "Con b<0 o d<0 la nube se refleja (el signo de ρ cambia)."
                )

    # ══════════════════════════════════════════════════════════════════════
    # EJERCICIO 5 — NORMAL MULTIVARIADA
    # ══════════════════════════════════════════════════════════════════════
    with ej5:
        st.markdown("### Ejercicio — Portafolio con retornos normales")
        st.markdown('<div class="ejemplo-box"><span class="label-ej">Enunciado</span>', unsafe_allow_html=True)
        st.markdown(
            "Sean $X_1$ y $X_2$ los retornos de dos activos con distribución normal. "
            "Se conocen los siguientes parámetros:"
        )
        col_en1, col_en2 = st.columns(2)
        with col_en1:
            st.latex(r"\mu_1 = 3\%,\quad \sigma_1 = 5\%")
            st.latex(r"\mu_2 = 7\%,\quad \sigma_2 = 12\%")
        with col_en2:
            st.latex(r"\text{Cov}(X_1,X_2) = -0.42\%^2")
            st.markdown("El activo 1 compone el **60%** del portafolio.")
        st.markdown(
            "**a)** Calcule el coeficiente de correlación e interprete.  \n"
            "**b)** Calcule la probabilidad de obtener un retorno positivo con el portafolio.  \n"
            "**c)** Dado que el primer activo tuvo un retorno del 2%, "
            "calcule el retorno esperado del portafolio."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### Resolución")

        # Parámetros
        mu1, mu2 = 3.0, 7.0
        s1, s2   = 5.0, 12.0
        cov12    = -0.42  # en %²... interpretado como -0.0042 en decimales o -0.42 en (%·%)
        w1, w2   = 0.60, 0.40

        st.markdown('<div class="section-title">Parte a) — Coeficiente de correlación</div>', unsafe_allow_html=True)
        st.markdown("**Fórmula:**")
        st.latex(r"\rho = \frac{\text{Cov}(X_1,X_2)}{\sigma_1\cdot\sigma_2}")
        st.markdown(
            "> **Nota sobre unidades:** los retornos están en porcentaje, "
            "entonces $\\sigma_1 = 5\\%$, $\\sigma_2 = 12\\%$, "
            "y la covarianza es $-0.42\\%^2$. "
            "Al dividir, las unidades se cancelan y $\\rho$ es adimensional."
        )
        rho_val = cov12 / (s1*s2)
        st.latex(
            r"\rho = \frac{-0.42\%^2}{5\%\times 12\%} = \frac{-0.42}{60} = -0.007"
        )

        st.markdown('<div class="obs-box"><span class="label-obs">Interpretación</span>', unsafe_allow_html=True)
        rho_val_real = -0.42/(5*12)
        st.markdown(
            f"$\\rho = {rho_val_real:.4f}$: correlación **negativa muy débil**.  \n"
            "Los retornos de los dos activos tienen una ligera tendencia a moverse en direcciones opuestas, "
            "pero la relación es prácticamente nula.  \n"
            "Esto sugiere que los activos ofrecen una pequeña diversificación entre sí."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte b) — Probabilidad de retorno positivo</div>', unsafe_allow_html=True)

        st.markdown("**Paso 1 — Definir el retorno del portafolio.**")
        st.latex(r"Y = w_1 X_1 + w_2 X_2 = 0.60\,X_1 + 0.40\,X_2")

        st.markdown(
            "**Paso 2 — Calcular $E[Y]$ y $\\text{Var}(Y)$.**  \n"
            "Por el Teorema 4 (combinaciones lineales de normales son normales):"
        )
        EY = w1*mu1 + w2*mu2
        VarY = w1**2*s1**2 + w2**2*s2**2 + 2*w1*w2*cov12
        sigY = np.sqrt(VarY)

        st.latex(
            r"E[Y] = 0.60\times 3\% + 0.40\times 7\% = 1.8\% + 2.8\% = 4.6\%"
        )
        st.latex(
            r"\text{Var}(Y) = (0.60)^2(5)^2 + (0.40)^2(12)^2 + 2(0.60)(0.40)(-0.42)"
        )
        st.latex(
            r"= 0.36\times 25 + 0.16\times 144 + 2(0.24)(-0.42)"
        )
        st.latex(
            r"= 9 + 23.04 - 0.2016 = 31.84\%^2"
        )
        st.latex(rf"\sigma_Y = \sqrt{{31.84}} \approx {sigY:.4f}\%")

        st.markdown("**Paso 3 — Usar normalidad de $Y$.**")
        st.latex(r"Y \sim N(4.6\%,\; 31.84\%^2)")
        st.markdown("**Paso 4 — Estandarizar y calcular la probabilidad:**")
        z_val = -EY/sigY
        prob_pos = 1 - stats.norm.cdf(z_val)

        st.latex(r"P(Y > 0) = P\!\left(Z > \frac{0-4.6}{" + f"{sigY:.4f}" + r"}\right) = P\!\left(Z > " + f"{z_val:.4f}" + r"\right)")
        st.latex(r"= 1 - \Phi(" + f"{z_val:.4f}" + r") = \Phi(" + f"{-z_val:.4f}" + r") \approx " + f"{prob_pos:.4f}")

        st.markdown('<div class="prop-box"><span class="label-prop">Resultado parte b)</span>', unsafe_allow_html=True)
        st.markdown(
            f"$P(Y > 0) \\approx {prob_pos:.4f} = {prob_pos*100:.2f}\\%$  \n\n"
            "Hay aproximadamente un **79% de probabilidad** de obtener un retorno positivo con este portafolio."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="section-title">Parte c) — Retorno esperado dado X₁ = 2%</div>', unsafe_allow_html=True)
        st.markdown(
            "Queremos $E[Y \\mid X_1 = 2\\%]$ donde $Y = 0.60X_1 + 0.40X_2$.  \n\n"
            "**Paso 1 — Usar linealidad de la esperanza condicional:**"
        )
        st.latex(r"E[Y \mid X_1=2] = 0.60\,E[X_1\mid X_1=2] + 0.40\,E[X_2\mid X_1=2]")
        st.latex(r"= 0.60\times 2 + 0.40\,E[X_2\mid X_1=2]")

        st.markdown(
            "**Paso 2 — Calcular $E[X_2 \\mid X_1 = 2]$ usando el Teorema 3 (condicional normal):**"
        )
        st.latex(
            r"E[X_2 \mid X_1=x_1] = \mu_2 + \frac{\sigma_2}{\sigma_1}\rho\,(x_1-\mu_1)"
        )
        rho_display = cov12/(s1*s2)
        EX2_cond = mu2 + (s2/s1)*rho_display*(2 - mu1)
        st.latex(
            r"= 7 + \frac{12}{5}\times(-0.007)\times(2-3)"
            r"= 7 + 2.4\times(-0.007)\times(-1)"
            r"= 7 + 0.0168 \approx 7.017\%"
        )

        st.markdown("**Paso 3 — Calcular $E[Y \\mid X_1=2]$:**")
        EY_cond = 0.60*2 + 0.40*EX2_cond
        st.latex(
            r"E[Y\mid X_1=2] = 0.60\times 2 + 0.40\times 7.017"
            r"= 1.2 + 2.807 = 4.007\%"
        )

        st.markdown('<div class="obs-box"><span class="label-obs">Interpretación</span>', unsafe_allow_html=True)
        st.markdown(
            f"Dado que el primer activo tuvo un retorno de 2% (por debajo de su media de 3%), "
            f"el retorno esperado del portafolio es $\\approx 4.0\\%$.  \n\n"
            "Este resultado es ligeramente inferior al retorno esperado incondicional $E[Y] = 4.6\\%$.  \n"
            "¿Por qué? Porque $\\rho < 0$: cuando $X_1$ está por debajo de su media, "
            "esperamos que $X_2$ esté **ligeramente por encima** de la suya (efecto de la correlación negativa), "
            "lo que atenúa la caída del portafolio.  \n"
            "Sin embargo, el efecto es muy pequeño dado que $|\\rho|$ es casi cero."
        )
        st.markdown("</div>", unsafe_allow_html=True)

        # Visualización: distribución del portafolio
        st.markdown("---")
        st.markdown("#### Visualización — Distribución del retorno del portafolio")
        col_vn_e5a, col_vn_e5b = st.columns([1,2])
        with col_vn_e5a:
            st.markdown("**Parámetros del portafolio:**")
            st.markdown(f"$E[Y] = {EY:.2f}\\%$")
            st.markdown(f"$\\sigma_Y = {sigY:.4f}\\%$")
            st.markdown(f"$P(Y>0) \\approx {prob_pos*100:.2f}\\%$")
            st.markdown("---")
            st.markdown(
                "La línea vertical roja marca el punto de retorno cero. "
                "El área sombreada a la derecha es la probabilidad buscada."
            )
        with col_vn_e5b:
            y_grid_e5 = np.linspace(EY-4*sigY, EY+4*sigY, 400)
            f_e5 = stats.norm(EY, sigY).pdf(y_grid_e5)
            y_pos = y_grid_e5[y_grid_e5 >= 0]
            f_pos = stats.norm(EY, sigY).pdf(y_pos)

            fig_e5 = go.Figure()
            fig_e5.add_trace(go.Scatter(
                x=y_grid_e5, y=f_e5, mode='lines',
                name=f'Y ~ N({EY:.1f}, {VarY:.2f})',
                line=dict(color='#2196F3', width=3)
            ))
            fig_e5.add_trace(go.Scatter(
                x=np.concatenate([[0], y_pos, [y_pos[-1]]]),
                y=np.concatenate([[0], f_pos, [0]]),
                fill='toself',
                fillcolor='rgba(76,175,80,0.3)',
                line=dict(color='rgba(0,0,0,0)'),
                name=f'P(Y>0) = {prob_pos:.3f}'
            ))
            fig_e5.add_vline(x=0, line_dash="dash", line_color="#e63946", line_width=2,
                             annotation_text="Y=0", annotation_font_color="#e63946")
            fig_e5.add_vline(x=EY, line_dash="dot", line_color="white", opacity=0.5,
                             annotation_text=f"E[Y]={EY:.1f}%", annotation_font_color="white")
            fig_e5.update_layout(
                xaxis_title="Retorno del portafolio Y (%)",
                yaxis_title="Densidad",
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='white', height=380,
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(font=dict(color='white'))
            )
            fig_e5.update_xaxes(gridcolor='#222')
            fig_e5.update_yaxes(gridcolor='#222')
            st.plotly_chart(fig_e5, use_container_width=True)
            st.caption(
                f"La curva muestra Y ~ N({EY:.1f}%, {VarY:.2f}%²).  "
                f"El área verde es P(Y>0) ≈ {prob_pos*100:.1f}%.  "
                "La distribución está centrada a la derecha del cero — "
                "es más probable obtener retorno positivo que negativo."
            )

