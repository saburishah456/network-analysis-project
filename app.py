import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from collections import Counter

st.set_page_config(page_title="LinkedIn Network Analysis Dashboard", layout="wide")

# ─────────────────────────────────────────────
# THEME COLORS
# ─────────────────────────────────────────────
COMPANY_COLOR   = "#AFA9EC"
COMPANY_EDGE    = "#534AB7"
SKILL_COLOR     = "#5DCAA5"
SKILL_EDGE      = "#0F6E56"
COMM_COLORS     = ["#534AB7", "#1D9E75", "#BA7517", "#D85A30", "#993556"]
BAR_COLOR       = "#534AB7"
BAR_COLOR2      = "#1D9E75"

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_json(
        "linkedin-jobs__20230801_20230815_sample.ldjson",
        lines=True
    )

    # Curated skill keywords — extracted properly from descriptions
    SKILL_KEYWORDS = [
        "Python", "Java", "SQL", "Excel", "AWS", "React", "Docker",
        "Kubernetes", "C\\+\\+", "JavaScript", "Machine Learning",
        "Data Analysis", "Communication", "Management", "Leadership",
        "Marketing", "Finance", "Accounting", "PowerPoint", "Tableau",
        "Salesforce", "Git", "Linux", "Azure", "GCP", "TensorFlow",
        "PyTorch", "Scala", "Spark", "Hadoop", "MongoDB", "PostgreSQL",
        "Node.js", "R\\b"
    ]

    def extract_skills(text):
        found = []
        for kw in SKILL_KEYWORDS:
            if re.search(r"\b" + kw + r"\b", str(text), re.IGNORECASE):
                # Normalise display name
                clean = kw.replace("\\b", "").replace("\\+\\+", "++")
                found.append(clean)
        return found if found else ["Other"]

    df["skills"] = df["job_description"].fillna("").apply(extract_skills)
    df["company_name"] = df["company_name"].fillna("Unknown")
    df["category"]     = df.get("category",     pd.Series([""] * len(df))).fillna("Unknown")
    df["seniority_level"] = df.get("seniority_level", pd.Series([""] * len(df))).fillna("Not specified")
    df["is_remote"]    = df.get("is_remote",    pd.Series([False] * len(df))).fillna(False)
    df["job_type"]     = df.get("job_type",     pd.Series([""] * len(df))).fillna("Unknown")
    df["state"]        = df.get("state",        pd.Series([""] * len(df))).fillna("Unknown")
    df["job_title"]    = df.get("job_title",    pd.Series([""] * len(df))).fillna("Unknown")
    df["test_educational_credential"] = df.get(
        "test_educational_credential", pd.Series([""] * len(df))
    ).fillna("Not specified")

    return df


df = load_data()

# ─────────────────────────────────────────────
# BUILD BIPARTITE GRAPH (company ↔ skill)
# ─────────────────────────────────────────────
@st.cache_data
def build_graph(_df):
    G = nx.Graph()
    for _, row in _df.iterrows():
        for skill in row["skills"]:
            if skill != "Other":
                G.add_edge(row["company_name"], skill,
                           weight=G[row["company_name"]][skill]["weight"] + 1
                           if G.has_edge(row["company_name"], skill) else 1)
    return G

G = build_graph(df)

# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "📂 Dataset",
    "📊 Overview",
    "📈 Skill Trends",
    "🔗 Centrality Analysis",
    "🧩 Structural Holes",
    "🌐 Network Graph",
    "🎯 Community Detection",
])

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def bar_chart(labels, values, title, color=BAR_COLOR, xlabel="Count", figsize=(7, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(labels[::-1], values[::-1], color=color, edgecolor="white", linewidth=0.6)
    ax.bar_label(bars, padding=4, fontsize=9, color="#444")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=10)
    plt.tight_layout()
    return fig


# ─────────────────────────────────────────────
# 📂 DATASET
# ─────────────────────────────────────────────
if section == "📂 Dataset":
    st.header("Dataset Preview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total postings",    len(df))
    col2.metric("Unique companies",  df["company_name"].nunique())
    col3.metric("Remote roles",      int(df["is_remote"].sum()))
    col4.metric("Full-time roles",   int((df["job_type"] == "Full-time").sum()))

    st.divider()
    display_cols = ["job_title", "company_name", "state", "job_type",
                    "category", "is_remote", "seniority_level"]
    st.dataframe(
        df[display_cols].rename(columns={
            "job_title": "Title",
            "company_name": "Company",
            "state": "State",
            "job_type": "Type",
            "category": "Category",
            "is_remote": "Remote",
            "seniority_level": "Seniority",
        }),
        use_container_width=True,
        height=380,
    )


# ─────────────────────────────────────────────
# 📊 OVERVIEW
# ─────────────────────────────────────────────
elif section == "📊 Overview":
    st.header("Overview")

    # Skill frequency
    all_skills = df["skills"].explode().loc[lambda x: x != "Other"]
    skill_counts = all_skills.value_counts()

    # Category counts
    cat_counts = df["category"].value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top skills across all postings")
        fig = bar_chart(
            skill_counts.head(10).index.tolist(),
            skill_counts.head(10).values.tolist(),
            "Top 10 Skills",
            color=BAR_COLOR2,
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Job categories")
        fig = bar_chart(
            cat_counts.index.tolist(),
            cat_counts.values.tolist(),
            "Postings by Category",
            color=BAR_COLOR,
        )
        st.pyplot(fig)
        plt.close()

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Job type distribution")
        jt = df["job_type"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5))
        wedges, texts, autotexts = ax.pie(
            jt.values,
            labels=jt.index,
            autopct="%1.0f%%",
            colors=["#AFA9EC", "#5DCAA5", "#FAC775"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        for at in autotexts:
            at.set_fontsize(11)
        ax.set_title("Job Types", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col4:
        st.subheader("Remote vs on-site")
        remote = df["is_remote"].value_counts().rename({True: "Remote", False: "On-site"})
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.pie(
            remote.values,
            labels=remote.index,
            autopct="%1.0f%%",
            colors=["#5DCAA5", "#B4B2A9"],
            startangle=90,
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title("Remote vs On-site", fontsize=12, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# 📈 SKILL TRENDS
# ─────────────────────────────────────────────
elif section == "📈 Skill Trends":
    st.header("Skill Trends")

    all_skills = df["skills"].explode().loc[lambda x: x != "Other"]
    skill_counts = all_skills.value_counts()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("All skills by frequency")
        fig = bar_chart(
            skill_counts.index.tolist(),
            skill_counts.values.tolist(),
            "Skill Frequency",
            color=BAR_COLOR2,
            figsize=(7, max(4, len(skill_counts) * 0.45)),
        )
        st.pyplot(fig)
        plt.close()

    with col2:
        st.subheader("Seniority level distribution")
        sen = df["seniority_level"].value_counts()
        fig = bar_chart(
            sen.index.tolist(), sen.values.tolist(),
            "Seniority Levels", color=BAR_COLOR,
        )
        st.pyplot(fig)
        plt.close()

        st.subheader("Education requirements")
        edu = df["test_educational_credential"].value_counts()
        fig = bar_chart(
            edu.index.tolist(), edu.values.tolist(),
            "Education Requirements", color="#D85A30",
        )
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.subheader("Top companies by posting count")
    comp = df["company_name"].value_counts()
    fig = bar_chart(
        comp.head(10).index.tolist(),
        comp.head(10).values.tolist(),
        "Top Companies", color="#993556",
    )
    st.pyplot(fig)
    plt.close()


# ─────────────────────────────────────────────
# 🔗 CENTRALITY ANALYSIS
# ─────────────────────────────────────────────
elif section == "🔗 Centrality Analysis":
    st.header("Centrality Analysis")
    st.caption(
        "Nodes are companies and skills. "
        "An edge connects a company to a skill it requires. "
        "Centrality measures influence within the network."
    )

    tab1, tab2, tab3 = st.tabs(["Degree", "Betweenness", "Closeness"])

    def top_centrality(metric_dict, n=12):
        top = sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:n]
        labels = [t[0] for t in top]
        values = [round(t[1], 4) for t in top]
        return labels, values

    with tab1:
        st.subheader("Degree centrality")
        st.caption("Nodes with the most direct connections — highly demanded skills or well-connected companies.")
        labels, values = top_centrality(nx.degree_centrality(G))
        fig = bar_chart(labels, values, "Degree Centrality (top 12)", color=BAR_COLOR, xlabel="Score")
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Betweenness centrality")
        st.caption("Nodes that act as bridges between parts of the network.")
        labels, values = top_centrality(nx.betweenness_centrality(G))
        fig = bar_chart(labels, values, "Betweenness Centrality (top 12)", color=BAR_COLOR2, xlabel="Score")
        st.pyplot(fig)
        plt.close()

    with tab3:
        st.subheader("Closeness centrality")
        st.caption("Nodes that can reach all others in the fewest hops.")
        labels, values = top_centrality(nx.closeness_centrality(G))
        fig = bar_chart(labels, values, "Closeness Centrality (top 12)", color="#D85A30", xlabel="Score")
        st.pyplot(fig)
        plt.close()


# ─────────────────────────────────────────────
# 🧩 STRUCTURAL HOLES
# ─────────────────────────────────────────────
elif section == "🧩 Structural Holes":
    st.header("Structural Holes")
    st.caption(
        "A low **constraint** score means a node bridges otherwise disconnected clusters — "
        "a structural broker. High constraint = embedded in a tight clique."
    )

    from networkx.algorithms.structuralholes import constraint
    c_values = constraint(G)

    top = sorted(c_values.items(), key=lambda x: x[1])[:12]
    labels = [t[0] for t in top]
    values = [round(t[1], 4) for t in top]

    fig = bar_chart(
        labels, values,
        "Structural Holes — lowest constraint (brokers) first",
        color="#BA7517", xlabel="Constraint score (lower = more brokerage)",
        figsize=(8, 5),
    )
    st.pyplot(fig)
    plt.close()

    with st.expander("What does this mean?"):
        st.markdown(
            "- **Low constraint** nodes span different parts of the network, "
            "giving them information and brokerage advantages.\n"
            "- **High constraint** nodes are embedded in dense clusters where "
            "all their neighbours already know each other.\n"
            "- In a job-market context, skills that bridge multiple industries "
            "or companies are the most strategically valuable."
        )


# ─────────────────────────────────────────────
# 🌐 NETWORK GRAPH
# ─────────────────────────────────────────────
elif section == "🌐 Network Graph":
    st.header("Network Graph")

    tab1, tab2 = st.tabs(["Company ↔ Skill bipartite", "Skill co-occurrence"])

    with tab1:
        st.subheader("Company ↔ Skill bipartite network")
        st.caption(
            "**Purple nodes** = companies · **Teal nodes** = skills · "
            "Edge thickness ∝ how many times a skill appears for that company."
        )

        top_n_skills = st.slider("Max skills to display", 5, 20,
                                 value=min(12, len(df["skills"].explode().unique())), key="bip_slider")

        all_skills = df["skills"].explode().loc[lambda x: x != "Other"]
        top_skills_set = set(all_skills.value_counts().head(top_n_skills).index)

        # Subgraph
        kept_nodes = set(df["company_name"]) | top_skills_set
        sub = G.subgraph([n for n in G.nodes if n in kept_nodes])

        # Bipartite layout: companies left, skills right
        companies = list(df["company_name"].unique())
        skills_in_sub = [n for n in sub.nodes if n not in companies]

        pos = {}
        for i, c in enumerate(companies):
            pos[c] = (-1, 1 - 2 * i / max(len(companies) - 1, 1))
        for i, s in enumerate(skills_in_sub):
            pos[s] = (1, 1 - 2 * i / max(len(skills_in_sub) - 1, 1))

        node_colors = [COMPANY_COLOR if n in companies else SKILL_COLOR for n in sub.nodes]
        node_sizes  = [400 if n in companies else 200 + all_skills.value_counts().get(n, 1) * 80
                       for n in sub.nodes]

        weights = [sub[u][v].get("weight", 1) for u, v in sub.edges]
        max_w   = max(weights) if weights else 1
        widths  = [0.5 + 2.5 * w / max_w for w in weights]

        fig, ax = plt.subplots(figsize=(12, 8))
        nx.draw_networkx_edges(sub, pos, ax=ax, width=widths, alpha=0.35, edge_color="#888780")
        nx.draw_networkx_nodes(sub, pos, ax=ax,
                               node_color=node_colors, node_size=node_sizes,
                               linewidths=1.5,
                               edgecolors=[COMPANY_EDGE if n in companies else SKILL_EDGE for n in sub.nodes])
        nx.draw_networkx_labels(sub, pos, ax=ax, font_size=9, font_color="#2C2C2A")
        ax.axis("off")
        legend_handles = [
            mpatches.Patch(color=COMPANY_COLOR, label="Company", linewidth=1),
            mpatches.Patch(color=SKILL_COLOR,   label="Skill",   linewidth=1),
        ]
        ax.legend(handles=legend_handles, loc="lower right", frameon=True, fontsize=10)
        plt.title("Company ↔ Skill Bipartite Network", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        st.subheader("Skill co-occurrence network")
        st.caption(
            "An edge connects two skills that appear in **the same job posting**. "
            "Edge thickness ∝ co-occurrence frequency. Node size ∝ skill frequency."
        )

        top_n_co = st.slider("Max skills to display", 5, 20, 10, key="co_slider")
        top_skills_list = all_skills.value_counts().head(top_n_co).index.tolist()

        skill_graph = nx.Graph()
        for skills_list in df["skills"]:
            filtered = [s for s in skills_list if s in top_skills_list]
            for i in range(len(filtered)):
                for j in range(i + 1, len(filtered)):
                    a, b = filtered[i], filtered[j]
                    if skill_graph.has_edge(a, b):
                        skill_graph[a][b]["weight"] += 1
                    else:
                        skill_graph.add_edge(a, b, weight=1)

        if skill_graph.number_of_nodes() == 0:
            st.info("Not enough co-occurring skills with current selection.")
        else:
            pos = nx.spring_layout(skill_graph, seed=42, k=1.5)
            node_sizes = [200 + all_skills.value_counts().get(n, 1) * 120 for n in skill_graph.nodes]
            edge_weights = [skill_graph[u][v]["weight"] for u, v in skill_graph.edges]
            max_w = max(edge_weights) if edge_weights else 1

            fig, ax = plt.subplots(figsize=(9, 7))
            nx.draw_networkx_edges(
                skill_graph, pos, ax=ax,
                width=[0.5 + 3 * w / max_w for w in edge_weights],
                alpha=0.5, edge_color="#B4B2A9",
            )
            nx.draw_networkx_nodes(
                skill_graph, pos, ax=ax,
                node_color=SKILL_COLOR, node_size=node_sizes,
                edgecolors=SKILL_EDGE, linewidths=1.5,
            )
            nx.draw_networkx_labels(skill_graph, pos, ax=ax, font_size=9, font_color="#2C2C2A")
            ax.axis("off")
            plt.title("Skill Co-occurrence Network", fontsize=14, fontweight="bold")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()


# ─────────────────────────────────────────────
# 🎯 COMMUNITY DETECTION
# ─────────────────────────────────────────────
elif section == "🎯 Community Detection":
    st.header("Community Detection")
    st.caption(
        "Communities are detected on the **skill co-occurrence graph** "
        "using the greedy modularity algorithm. Each colour = one community."
    )

    from networkx.algorithms import community as nx_community

    all_skills = df["skills"].explode().loc[lambda x: x != "Other"]
    top_n = st.slider("Number of top skills to include", 5, 20, 12)
    top_skills_list = all_skills.value_counts().head(top_n).index.tolist()

    skill_graph = nx.Graph()
    for skills_list in df["skills"]:
        filtered = [s for s in skills_list if s in top_skills_list]
        for i in range(len(filtered)):
            for j in range(i + 1, len(filtered)):
                a, b = filtered[i], filtered[j]
                if skill_graph.has_edge(a, b):
                    skill_graph[a][b]["weight"] += 1
                else:
                    skill_graph.add_edge(a, b, weight=1)

    if skill_graph.number_of_nodes() < 2:
        st.info("Not enough nodes. Try increasing the number of top skills.")
    else:
        communities = list(nx_community.greedy_modularity_communities(skill_graph))

        # Build colour map
        color_map = {}
        for i, comm in enumerate(communities):
            for node in comm:
                color_map[node] = COMM_COLORS[i % len(COMM_COLORS)]

        node_colors = [color_map.get(n, "#B4B2A9") for n in skill_graph.nodes]
        pos = nx.spring_layout(skill_graph, seed=42, k=1.5)
        node_sizes = [300 + all_skills.value_counts().get(n, 1) * 100 for n in skill_graph.nodes]

        fig, ax = plt.subplots(figsize=(9, 7))
        edge_weights = [skill_graph[u][v]["weight"] for u, v in skill_graph.edges]
        max_w = max(edge_weights) if edge_weights else 1
        nx.draw_networkx_edges(
            skill_graph, pos, ax=ax,
            width=[0.5 + 2.5 * w / max_w for w in edge_weights],
            alpha=0.4, edge_color="#B4B2A9",
        )
        nx.draw_networkx_nodes(
            skill_graph, pos, ax=ax,
            node_color=node_colors, node_size=node_sizes,
            edgecolors="white", linewidths=1.5,
        )
        nx.draw_networkx_labels(skill_graph, pos, ax=ax, font_size=9, font_color="#2C2C2A")

        legend_handles = [
            mpatches.Patch(color=COMM_COLORS[i], label=f"Community {i + 1}: {', '.join(list(c)[:4])}{'…' if len(c) > 4 else ''}")
            for i, c in enumerate(communities)
        ]
        ax.legend(handles=legend_handles, loc="lower right", fontsize=9, frameon=True)
        ax.axis("off")
        plt.title("Skill Community Detection", fontsize=14, fontweight="bold")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.divider()
        st.subheader("Detected communities")
        for i, comm in enumerate(communities):
            with st.expander(f"Community {i + 1} — {len(comm)} skill(s)"):
                st.write(", ".join(sorted(comm)))
