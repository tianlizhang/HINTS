import dgl
import torch
from tqdm import trange, tqdm

def calc_h_index(cites):
    c = torch.sort(cites, descending=True)[0]
    indx = torch.arange(len(c))
    delta = c-indx
    d = torch.where(delta>0)[0]
    if len(d)>0:
        h_index = d[-1]+1
    else:
        h_index = 0
    return h_index


def load():
    graph_list, _ = dgl.load_graphs('../save3/graph_vfc.graph')
    graph = graph_list[0]

    author_ids = torch.where(graph.in_degrees(etype='written') >= 50)[0]
    paper_ids = torch.arange(graph.number_of_nodes('paper'))
    topic_ids = torch.arange(graph.number_of_nodes('topic'))
    venue_ids = torch.arange(graph.number_of_nodes('venue'))
    field_ids = torch.arange(graph.number_of_nodes('field'))
    org_ids = torch.arange(graph.number_of_nodes('org'))
    country_ids = torch.arange(graph.number_of_nodes('country'))

    g = graph.subgraph({'author': author_ids, 'paper': paper_ids, 'topic':topic_ids, \
        'venue': venue_ids, 'field': field_ids, 'org':org_ids, 'country': country_ids})
    print(g)
    return g


def process(graph, year=2000):
    g = dgl.node_type_subgraph(graph, ['paper', 'venue', 'author', 'topic', 'field'])

    written_eids = g.filter_edges(lambda x: x.data['year'] >= year, etype='written')
    contains_eids = g.filter_edges(lambda x: x.data['year'] >= year, etype='contains')
    publishes_eids = g.filter_edges(lambda x: x.data['year'] >= year, etype='publishes')
    has_eids = g.filter_edges(lambda x: x.data['year'] >= year, etype='has')
    cites_eids = g.filter_edges(lambda x: x.data['year'] >= year, etype='cites')

    year_graph = g.edge_subgraph({'written':written_eids, 'contains': contains_eids, \
        'publishes': publishes_eids, 'has': has_eids, 'cites': cites_eids}, preserve_nodes=False)
    return year_graph


def limit_year(g, year):
    written_eids = g.filter_edges(lambda x: x.data['year'] <= year, etype='written')
    contains_eids = g.filter_edges(lambda x: x.data['year'] <= year, etype='contains')
    publishes_eids = g.filter_edges(lambda x: x.data['year'] <= year, etype='publishes')
    has_eids = g.filter_edges(lambda x: x.data['year'] <= year, etype='has')
    cites_eids = g.filter_edges(lambda x: x.data['year'] <= year, etype='cites')

    year_graph = g.edge_subgraph({'written':written_eids, 'contains': contains_eids, \
        'publishes': publishes_eids, 'has': has_eids, 'cites': cites_eids}, preserve_nodes=False)
    return year_graph


def add(graph):
    in_degree = graph.in_degrees(etype='cites')
    graph.nodes['paper'].data['citation'] = in_degree
    citation = graph.nodes['paper'].data['citation']

    def add_h_index(name='topic', etype='contains', is_paper_dst=True):
        nodes = graph.nodes(name)
        source, target = graph.edges(etype = etype)

        h_index = torch.zeros(len(nodes))
        outer = range(len(nodes))
        for i in outer:
            if is_paper_dst==True:
                idx = torch.where(source==nodes[i])[0]
                cite_ids = target[idx]
            else:
                idx = torch.where(target==nodes[i])[0]
                cite_ids = source[idx]

            cites = citation[cite_ids]
            h_index[i] = calc_h_index(cites)
        graph.nodes[name].data['h_index'] = h_index
        max_id = torch.max(h_index)
        print(f'{name}: {graph.nodes[name]}, {max_id}')
        # outer.set_postfix(name=name)

    add_h_index('topic', 'contains', is_paper_dst=True)
    add_h_index('author', 'written', is_paper_dst=False)
    add_h_index('venue', 'publishes', is_paper_dst=True)
    return graph


if __name__ == '__main__':
    # graph = load()
    # graph = add(graph)
    # dgl.save_graphs('../save3/graph_h_index.graph', [graph])

    graph = load()
    g = process(graph, 2000)
    out = []
    for year in trange(2000, 2022):
        year_graph = limit_year(g, year)
        print(f'{year:}, {year_graph}')
        new_g = add(year_graph)
        out.append(new_g)
    dgl.save_graphs('../save3/year_graphs_h_index.graph', out)