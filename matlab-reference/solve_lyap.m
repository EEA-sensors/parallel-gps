% Solve Lyapunov equation via vectorization
function Pinf = solve_lyap(F,L,q)
    dim = size(F,1);
    F1 = kron(eye(dim),F);
    F2 = kron(F,eye(dim));
    Q  = L * q * L';
    Pinf = reshape(-(F1 + F2)\Q(:),dim,dim);
    Pinf = 0.5 * (Pinf + Pinf');
end

