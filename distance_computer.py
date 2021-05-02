from torch import load
import os

class DistanceComputer():
    def __init__(self, hier_data_path):
        self.hier_data_path = hier_data_path
        self.parent2child = load(os.path.join(self.hier_data_path, 'parent2children_tree.p'))
        self.child2parent = load(os.path.join(self.hier_data_path, 'child2parent.p'))

    def compute_distance(self, code1, code2):
        def get_ancestors(child, anc):
            try:
                parent = self.child2parent[child]
                parent = parent[0] if type(parent) == list else parent
                anc.append(parent)
                child = parent
                get_ancestors(child, anc)
            except:
                pass

        ancestors1 = []
        get_ancestors(code1, ancestors1)
        ancestors2 = []
        get_ancestors(code2, ancestors2)

        if len(ancestors1) < len(ancestors2):
            ancestors1 = ['Null'] * (len(ancestors2) - len(ancestors1)) + ancestors1
        elif len(ancestors2) < len(ancestors1):
            ancestors2 = ['Null'] * (len(ancestors1) - len(ancestors2)) + ancestors2
        is_ancestor = False
        for i, (anc1, anc2) in enumerate(zip(ancestors1[::-1], ancestors2[::-1])):
            if anc1 == anc2:
                lca = anc1
                continue
            elif anc1 in {code1, code2}:
                lca = anc1
                is_ancestor = True
                break
            elif anc2 in {code1, code2}:
                lca = anc2
                is_ancestor = True
                break
            else:
                if lca != 'root':
                    ancestors1, ancestors2 = ancestors1[:-i+1], ancestors2[:-i+1]
                break
        if lca in ancestors1[:-1] and lca in ancestors2[:-1]:
            # the codes are siblings
            ancestors1, ancestors2 = [lca], [lca]
        if is_ancestor:
            try:
                ancestors1 = [a for a in ancestors1[:ancestors1.index(lca)+1] if a != 'Null']
            except:
                ancestors1 = []
            try:
                ancestors2 = [a for a in ancestors2[:ancestors2.index(lca)+1] if a != 'Null']
            except:
                ancestors2 = []

        ancestors1 = list(filter(lambda x: x !='Null', ancestors1))
        ancestors2 = list(filter(lambda x: x !='Null', ancestors2))

        distance = len(ancestors1) + len(ancestors2)
        # print(code1, ancestors1)
        # print(code2, ancestors2)
        # print(distance)
        return distance






if __name__=='__main__':
    dc = DistanceComputer('data/hierarchical_data/es/')
    # dc = DistanceComputer('data/hierarchical_data/cantemist/')

    # dc.compute_distance('V01-X59', 'X60-X84')
    # dc.compute_distance('a01.4', 'z96.619')
    # print('-'*25)
    dc.compute_distance('z96.619', 'z96.612')
    # dc.find_lca('a01.4', 'a17.82')
    # dc.compute_distance('8001/1', '9985/3')