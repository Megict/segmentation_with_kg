import plotly.graph_objects as go
import networkx as nx
import numpy as np

def draw_graph(G, pos, links = [],
                       link_color_key = {},
                       display_edges = True,
                       highlight_around = [], # обязательно оставить 
                       edge_limit_key_name = None, edge_limit_key_values = [0, 0], # показвать только грани со значениями в пределах
                       color_key = None, show_all = False):
    
    print("-------------------")
    print(highlight_around)
    print(color_key)
    print(show_all)
    print(edge_limit_key_name)
    print(edge_limit_key_values)
    print("-------------------")
    
    if highlight_around != []:
        edge_limit_key_name = None

    for node in G.nodes:
        #проверяем, заданы ли необходимые параметры узлов и задаем сами, если нет
        G.nodes[node].setdefault("label", node)
        G.nodes[node].setdefault("show", True)
        G.nodes[node].setdefault(color_key, 0)

    hl_out_edge_x = []
    hl_out_edge_y = []

    hl_in_edge_x = []
    hl_in_edge_y = []
        
    edge_x = []
    edge_y = []

    edge_text = []
    edge_opacity = []
    nodes_hold = [] + highlight_around # это узлы, которые нельзя делать прозрачными

    # -------------------------------------------------------
    # обработка ребер

    if display_edges:
        for edge in G.edges():
            if not edge_limit_key_name is None: # это только для позиции в тексте пока что
                if G.edges()[edge][edge_limit_key_name][0]['p'] < edge_limit_key_values[0] or G.edges()[edge][edge_limit_key_name][0]['p'] > edge_limit_key_values[1]:
                    continue
                else:
                    if edge[0] not in nodes_hold:
                        nodes_hold.append(edge[0])
                    if edge[1] not in nodes_hold:
                        nodes_hold.append(edge[1])
            
            x0, y0 = pos[edge[1]]
            x1, y1 = pos[edge[0]]
            
            if G.nodes[edge[1]]["label"] in highlight_around: 
                hl_out_edge_x.append(x0)
                hl_out_edge_x.append(x1)
                hl_out_edge_x.append(None)
                hl_out_edge_y.append(y0)
                hl_out_edge_y.append(y1)
                hl_out_edge_y.append(None)

                if edge[0] not in nodes_hold:
                    nodes_hold.append(edge[0])
                continue

            if G.nodes[edge[0]]["label"] in highlight_around:
                hl_in_edge_x.append(x0)
                hl_in_edge_x.append(x1)
                hl_in_edge_x.append(None)
                hl_in_edge_y.append(y0)
                hl_in_edge_y.append(y1)
                hl_in_edge_y.append(None)

                if not edge[1] in nodes_hold:
                    nodes_hold.append(edge[1])
                continue

            edge_x.append([x0, x1, None])
            edge_y.append([y0, y1, None])

            # edge_text.append(G.edges()[edge]['locations'])
            edge_text.append(G.edges()[edge]['label'][1] + ' ' + str(G.edges()[edge]['locations'][0]['p']) + ' ' +\
                                                        '(' + str(len(G.edges()[edge]['locations'])) + ')')
            edge_opacity.append(len(G.edges()[edge]['locations']))

        om = max(edge_opacity)
        edge_opacity = [o / om for o in edge_opacity]

    edge_traces = []

    for i in range(len(edge_x)):
        edge_traces.append(go.Scatter(
            x=edge_x[i], y=edge_y[i],
            hoverinfo='none',
            opacity = max(edge_opacity[i], 0.2) if highlight_around == [] else 0.2,
            mode="lines+markers" if nx.is_directed(G) else "lines",
            line=dict(
                width=0.5,
                color = "#888"
            ),
            marker=dict(
                symbol="arrow",
                size=15,
                angleref="previous",
            )))
    
        edge_traces.append(go.Scatter(
            x = [(edge_x[i][k*3] + edge_x[i][k*3 + 1]) / 2  for k in range(int(len(edge_x[i]) / 3))], y = [(edge_y[i][k*3] + edge_y[i][k*3 + 1]) / 2  for k in range(int(len(edge_y[i]) / 3))],
            hoverinfo='text',
            opacity = 0,
            mode="markers+text",
            text = edge_text[i],
            marker=dict(
                size=5,
                color = "white"
            )))
    
    hl_out_edge_trace = go.Scatter(
        x=hl_out_edge_x, y=hl_out_edge_y,
        hoverinfo='none',
        mode="lines+markers" if nx.is_directed(G) else "lines",
        line=dict(width=0.5, color="#800"),
        marker=dict(
            symbol="arrow",
            size=15,
            angleref="previous",
        ))
    
    hl_in_edge_trace = go.Scatter(
        x=hl_in_edge_x, y=hl_in_edge_y,
        hoverinfo='none',
        mode="lines+markers" if nx.is_directed(G) else "lines",
        line=dict(width=0.5, color="blue"),
        marker=dict(
            symbol="arrow",
            size=15,
            angleref="previous",
        ))
    # -------------------------------------------------------
    # обработка связей

    links_x = []
    links_y = []
    link_text = []
    for link in links:
        if link[0] in G.nodes() and link[1] in G.nodes():
            x0, y0 = pos[link[1]]
            x1, y1 = pos[link[0]]
            links_x.append([x0,x1,None])
            links_y.append([y0,y1,None])
            link_text.append(links[link]["type"])

    link_traces = []
    for i in range(len(links_x)):
        link_traces.append(go.Scatter(
            x=links_x[i], y=links_y[i],
            hoverinfo='none',
            mode="lines",
            opacity = 0.9,
            line=dict(width=0.5, color=link_color_key[link_text[i]]),
            marker=dict(
                symbol="arrow",
                size=15,
                angleref="previous",
            )))
    # -------------------------------------------------------
    # обработка вершин
    node_x = []
    node_y = []
    node_text = []
    node_hovertext = []
    node_colorscale = []
    node_opacity = []

    for node in G.nodes:
        if not show_all and G.nodes[node]["show"] == False:
            continue

        if color_key != None and G.nodes[node][color_key] == -1:
            G.nodes[node][color_key] = np.inf

        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        if nodes_hold == [] or node in nodes_hold:
            cur_hovertext = f'{node}\n'# {len(G.nodes[node]["locations"])} ({G.nodes[node]["locations"]})'
            node_text.append(f'{node}')
            node_hovertext.append(cur_hovertext + " (" + str(len(G.nodes[node]["locations"])) + ")" if len(cur_hovertext) < 40 else cur_hovertext[0:37] + "...")
        else:
            node_text.append('')
            node_hovertext.append('')

        node_colorscale.append(G.nodes[node][color_key] if color_key != None else 0)

        if nodes_hold == []:
            node_opacity.append(len(G.nodes[node]["locations"]))
        else:
            if node in nodes_hold:
                node_opacity.append(1)
            else:
                node_opacity.append(0.2)

    if nodes_hold == []:
        om = max(node_opacity)
        for i in range(len(node_opacity)):
            node_opacity[i] /= om

    # ------------------------------------------------------------------------------------
    node_traces = []
    for i in range(len(node_x)):
        node_traces.append(go.Scatter(
            x=[node_x[i]], y=[node_y[i]],
            mode= 'markers+text',
            text = node_text[i],
            hovertext = node_hovertext[i],
            hoverinfo='text',
            opacity = max(node_opacity[i], 0.2),

            marker=dict(
                showscale=True,
                # colorscale options
                #'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
                #'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
                #'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
                colorscale='Rainbow' if color_key == 'subset' else 'RdBu' if color_key == 'dp' else 'deep',
                reversescale=True,
                color = node_colorscale[i],
                size = 10,
                line_width=2)
            )
        )
        
    fig = go.Figure(data = edge_traces + [hl_in_edge_trace, hl_out_edge_trace] + link_traces + node_traces,
                    layout=go.Layout(
                        #title='<br>Граф связности терминов',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=30,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                        )
    fig.update_traces(textposition='top center')

    return fig
