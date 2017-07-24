filename = 'middle_layer.csv'
M = csvread(filename)

pkg load statistics
[COEFF SCORE LATENT] = princomp(zscore(M(:, 2: 31)))

A = find(M(:, 1) == 2)
B = find(M(:, 1) == 5)
C = find(M(:, 1) == 8)
D = find(M(:, 1) == 11)
E = find(M(:, 1) == 14)
F = find(M(:, 1) == 17)
G = find(M(:, 1) == 20)
H = find(M(:, 1) == 23)

figure('Position', [600, 300, 3000, 2000])
h = plot3(SCORE(A(1: length(A)), 1), SCORE(A(1: length(A)), 2), SCORE(A(1: length(A)), 3), 'r.', \
SCORE(B(1: length(B)), 1), SCORE(B(1: length(B)), 2), SCORE(B(1: length(B)), 3), 'g.', \
SCORE(C(1: length(C)), 1), SCORE(C(1: length(C)), 2), SCORE(C(1: length(C)), 3), 'b.', \
SCORE(D(1: length(D)), 1), SCORE(D(1: length(D)), 2), SCORE(D(1: length(D)), 3), 'y.', \
SCORE(E(1: length(E)), 1), SCORE(E(1: length(E)), 2), SCORE(E(1: length(E)), 3), 'm.', \
SCORE(F(1: length(F)), 1), SCORE(F(1: length(F)), 2), SCORE(F(1: length(F)), 3), 'c.', \
SCORE(G(1: length(G)), 1), SCORE(G(1: length(G)), 2), SCORE(G(1: length(G)), 3), 'k.', \
SCORE(H(1: length(H)), 1), SCORE(H(1: length(H)), 2), SCORE(H(1: length(H)), 3), 'bx')

%h = plot(SCORE(A(1: length(A)), 1), SCORE(A(1: length(A)), 2), 'r.', SCORE(B(1: length(B)), 1), SCORE(B(1: length(B)), 2), 'g.', SCORE(C(1: length(C)), 1), SCORE(C(1: length(C)), 2), 'b.')

pause
