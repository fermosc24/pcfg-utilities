import random
from collections import Counter, defaultdict

import numpy as np

from nltk import PCFG, Nonterminal,ProbabilisticProduction
from nltk.tree import Tree

from entropy_estimators import Entropy

def sample_tree(pcfg, symbol=None):
    """
    Iteratively sample a tree from a PCFG, using a stack (no recursion).
    Returns an NLTK Tree object.
    """
    if symbol is None:
        symbol = pcfg.start()
        
    # Start with the root node and a stack for expansion
    root = Tree(str(symbol), [])
    stack = [(root, symbol)]  # Each item: (tree_node, symbol_to_expand)
    
    while stack:
        node, sym = stack.pop()
        if isinstance(sym, Nonterminal):
            # Choose a production for this nonterminal
            productions = [p for p in pcfg.productions(lhs=sym)]
            probs = [p.prob() for p in productions]
            prod = random.choices(productions, weights=probs)[0]
            rhs = prod.rhs()
            # For each symbol in RHS, create child nodes if needed
            children = []
            for rhs_sym in rhs:
                if isinstance(rhs_sym, Nonterminal):
                    child = Tree(str(rhs_sym), [])
                    children.append(child)
                else:
                    children.append(rhs_sym)
            # Attach children to node
            node.extend(children)
            # Push children that are nonterminals to stack for further expansion, right-to-left
            for i in range(len(rhs)-1, -1, -1):
                if isinstance(rhs[i], Nonterminal):
                    stack.append((node[i], rhs[i]))
        # If sym is a terminal, nothing to do (already added as a string leaf)
    return root

def sample_n_trees(pcfg, n):
    """
    Sample n trees from the given PCFG.
    """
    return [sample_tree(pcfg) for _ in range(n)]

def induce_pcfg_from_trees(start, trees):
    """
    Induce a PCFG from a list of NLTK Tree objects.
    
    Args:
        start (str or Nonterminal): The start symbol.
        trees (list of nltk.Tree): Trees from which to extract productions.
        
    Returns:
        nltk.grammar.PCFG: The induced PCFG.
    """
    # Extract all productions from all trees
    all_productions = []
    for tree in trees:
        all_productions.extend(tree.productions())
    
    # Count productions
    prod_counts = Counter(all_productions)
    
    # Group productions by LHS for normalization
    lhs_counts = defaultdict(int)
    for prod, count in prod_counts.items():
        lhs_counts[prod.lhs()] += count
    
    # Make probabilistic productions
    prob_prods = []
    for prod, count in prod_counts.items():
        prob = count / lhs_counts[prod.lhs()]
        prob_prod = ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob)
        prob_prods.append(prob_prod)
    
    # Handle start symbol
    if not isinstance(start, Nonterminal):
        start = Nonterminal(start)
    
    # Create PCFG
    grammar = PCFG(start, prob_prods)
    return grammar

def characteristic_matrix(pcfg):
    """
    Returns:
        M: characteristic matrix (numpy array)
        nonterminals: list of nonterminals in order (row/column order in M)
    """
    nonterminals = sorted({prod.lhs() for prod in pcfg.productions()}, key=str)
    N = len(nonterminals)
    M = np.zeros((N, N))
    nt_to_idx = {nt: i for i, nt in enumerate(nonterminals)}
    for prod in pcfg.productions():
        i = nt_to_idx[prod.lhs()]
        p = prod.prob()
        counts = Counter(sym for sym in prod.rhs() if isinstance(sym, Nonterminal))
        for sym, count in counts.items():
            j = nt_to_idx[sym]
            M[i, j] += p * count
    return M, nonterminals

def induce_pcfg_with_counts(trees):
    all_productions = []
    for tree in trees:
        all_productions.extend(tree.productions())
    prod_counts = Counter(all_productions)
    lhs_to_rules = defaultdict(list)
    for prod, count in prod_counts.items():
        lhs_to_rules[prod.lhs()].append((prod, count))
    prob_prods = []
    for lhs, rules in lhs_to_rules.items():
        total = sum(count for _, count in rules)
        for prod, count in rules:
            prob = count / total
            prob_prods.append(ProbabilisticProduction(prod.lhs(), prod.rhs(), prob=prob))
    start = trees[0].label()
    grammar = PCFG(Nonterminal(start), prob_prods)
    return grammar, lhs_to_rules

def derivational_entropy(pcfg):
    """
    Compute derivational entropy for each nonterminal of a PCFG using rule probabilities.
    Returns a dict: {Nonterminal: derivational entropy}
    """
    M, nonterminals = characteristic_matrix(pcfg)
    nt_to_idx = {nt: i for i, nt in enumerate(nonterminals)}
    
    # 1. Compute h0: entropy of the rule probabilities for each nonterminal
    h0 = np.zeros(len(nonterminals))
    rules_by_lhs = defaultdict(list)
    for prod in pcfg.productions():
        rules_by_lhs[prod.lhs()].append(prod.prob())
    for idx, nt in enumerate(nonterminals):
        probs = rules_by_lhs[nt]
        if len(probs) > 1:
            h0[idx] = -sum(p * np.log2(p) for p in probs if p > 0)
        else:
            h0[idx] = 0.0

    # 2. Compute h = (I - M)^(-1) h0
    I = np.eye(M.shape[0])
    inv = np.linalg.inv(I - M)
    h = inv @ h0

    # 3. Map to dictionary
    result = {nt: float(h[idx]) for idx, nt in enumerate(nonterminals)}
    return result

def mean_length(pcfg):
    """
    Compute mean terminal yield length for each nonterminal in a PCFG.
    Returns dict: {Nonterminal: expected yield length}
    """
    M, nonterminals = characteristic_matrix(pcfg)
    nt_to_idx = {nt: i for i, nt in enumerate(nonterminals)}
    
    # Compute \ell_0 for each nonterminal
    ell_0 = np.zeros(len(nonterminals))
    for idx, nt in enumerate(nonterminals):
        # All productions for nt
        prods = [prod for prod in pcfg.productions(lhs=nt)]
        expected_terminals = 0.0
        for prod in prods:
            p = prod.prob()
            # Count terminals in RHS
            num_terminals = sum(1 for sym in prod.rhs() if not hasattr(sym, 'symbol'))
            expected_terminals += p * num_terminals
        ell_0[idx] = expected_terminals

    # Compute \ell = (I - M)^(-1) \ell_0
    I = np.eye(M.shape[0])
    inv = np.linalg.inv(I - M)
    ell = inv @ ell_0

    # Map to dictionary
    result = {nt: float(ell[idx]) for idx, nt in enumerate(nonterminals)}
    return result

def SITE(trees, method="MLE"):
    """
    Compute the Symbolic Information Transmission Entropy (SITE) for a collection of NLTK trees.
    Returns a dict: {Nonterminal: SITE value}
    """
    # 1. Induce PCFG and get rule counts
    pcfg, lhs_to_rules = induce_pcfg_with_counts(trees)
    # 2. Characteristic matrix and nonterminal order
    M, nonterminals = characteristic_matrix(pcfg)
    nt_to_idx = {nt: i for i, nt in enumerate(nonterminals)}
    # 3. h0: entropy for each nonterminal in order
    h0 = np.zeros(len(nonterminals))
    for idx, nt in enumerate(nonterminals):
        rule_counts = [count for prod, count in lhs_to_rules.get(nt, [])]
        if len(rule_counts) > 1:
            h0[idx] = Entropy(rule_counts, method)
        else:
            h0[idx] = 0.0
    # 4. h = (I-M)^(-1) h0
    I = np.eye(M.shape[0])
    inv = np.linalg.inv(I - M)
    h = inv @ h0
    # 5. Map to dictionary
    result = {nt: float(h[idx]) for idx, nt in enumerate(nonterminals)}
    return result

