% Almost the same as nchoosek()
function m = choose(n,k)

    % From codes of Solin/Särkkä:2014
    if all(size(n)==size(k))
        N = n;
        K = k;
    else
        K = repmat(k,[size(n,1) size(n,2)]);
        N = repmat(n,[size(k,1) size(k,2)]);
    end

    m = factorial(N)./factorial(K)./factorial(N-K);

end
