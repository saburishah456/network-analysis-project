import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

st.set_page_config(page_title="Network Analysis Dashboard", layout="wide")

st.title("📊 LinkedIn Network Analysis Dashboard")

# -------------------------
# LOAD DATASET (FIXED)
# -------------------------
df = pd.read_json("linkedin-jobs__20230801_20230815_sample.ldjson", lines=True)

# Fix column names (based on dataset)
df = df.rename(columns={
    "company": "company_name"
})

# Create "skills" from job description
df['skills'] = df['job_description'].fillna("").apply(lambda x: x.split()[:10])

df = df[['company_name', 'skills']].dropna()

# -------------------------
# BUILD GRAPH
# -------------------------
G = nx.Graph()

for _, row in df.iterrows():
    for skill in row['skills']:
        G.add_edge(row['company_name'], skill)

# -------------------------
# SIDEBAR
# -------------------------
section = st.sidebar.radio("Navigation", [
    "📂 Dataset",
    "📈 Trend Mining",
    "📊 Centrality Analysis",
    "🧩 Structural Holes",
    "🌐 Network Graph",
    "🎯 Community Detection"
])

# -------------------------
# DATASET
# -------------------------
if section == "📂 Dataset":
    st.subheader("Dataset Preview")
    st.dataframe(df.head(20))

# -------------------------
# TREND MINING
# -------------------------
elif section == "📈 Trend Mining":
    st.subheader("Top Skills")

    all_skills = df['skills'].explode()
    top_skills = all_skills.value_counts().head(10)

    fig, ax = plt.subplots()
    ax.bar(top_skills.index, top_skills.values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Skills")
    st.pyplot(fig)

    st.subheader("Top Companies")
    top_companies = df['company_name'].value_counts().head(10)

    fig, ax = plt.subplots()
    ax.bar(top_companies.index, top_companies.values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Top Companies")
    st.pyplot(fig)

# -------------------------
# CENTRALITY
# -------------------------
elif section == "📊 Centrality Analysis":

    tab1, tab2, tab3 = st.tabs(["Degree", "Betweenness", "Closeness"])

    with tab1:
        degree = nx.degree_centrality(G)
        top = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:10]

        labels = [i[0][:12] + "..." for i in top]
        values = [i[1] for i in top]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Degree Centrality")
        st.pyplot(fig)

    with tab2:
        between = nx.betweenness_centrality(G)
        top = sorted(between.items(), key=lambda x: x[1], reverse=True)[:10]

        labels = [i[0][:12] + "..." for i in top]
        values = [i[1] for i in top]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Betweenness Centrality")
        st.pyplot(fig)

    with tab3:
        close = nx.closeness_centrality(G)
        top = sorted(close.items(), key=lambda x: x[1], reverse=True)[:10]

        labels = [i[0][:12] + "..." for i in top]
        values = [i[1] for i in top]

        fig, ax = plt.subplots()
        ax.bar(labels, values)
        plt.xticks(rotation=45, ha='right')
        plt.title("Closeness Centrality")
        st.pyplot(fig)

# -------------------------
# STRUCTURAL HOLES
# -------------------------
elif section == "🧩 Structural Holes":

    from networkx.algorithms.structuralholes import constraint

    constraint_values = constraint(G)
    top = sorted(constraint_values.items(), key=lambda x: x[1])[:10]

    labels = [i[0][:12] + "..." for i in top]
    values = [i[1] for i in top]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    plt.xticks(rotation=45, ha='right')
    plt.title("Structural Hole (Low = Important)")
    st.pyplot(fig)

# -------------------------
# NETWORK GRAPH
# -------------------------
elif section == "🌐 Network Graph":

    st.subheader("Skill Network")

    all_skills = df['skills'].explode()
    top_skills = all_skills.value_counts().head(10)

    skill_graph = nx.Graph()

    for skills in df['skills']:
        for i in range(len(skills)):
            for j in range(i+1, len(skills)):
                skill_graph.add_edge(skills[i], skills[j])

    subgraph = skill_graph.subgraph(list(top_skills.index))

    pos = nx.spring_layout(subgraph, seed=42)

    fig, ax = plt.subplots(figsize=(6,5))
    nx.draw(subgraph, pos, with_labels=True, node_size=800, ax=ax)

    st.pyplot(fig)

# -------------------------
# COMMUNITY DETECTION
# -------------------------
elif section == "🎯 Community Detection":

    from networkx.algorithms import community

    skill_graph = nx.Graph()

    for skills in df['skills']:
        for i in range(len(skills)):
            for j in range(i+1, len(skills)):
                skill_graph.add_edge(skills[i], skills[j])

    communities = community.greedy_modularity_communities(skill_graph)

    top_skills = list(df['skills'].explode().value_counts().head(10).index)
    subgraph = skill_graph.subgraph(top_skills)

    pos = nx.spring_layout(subgraph, seed=42)

    color_map = {}
    for i, comm in enumerate(communities):
        for node in comm:
            color_map[node] = i

    node_colors = [color_map.get(node, 0) for node in subgraph.nodes()]

    fig, ax = plt.subplots(figsize=(6,5))
    nx.draw(subgraph, pos, with_labels=True, node_color=node_colors, node_size=800, ax=ax)

    st.pyplot(fig)
