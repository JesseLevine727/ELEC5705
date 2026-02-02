function mask = set_bins(mask, k0, guard, Knyq)
% Mark bins [k0-guard, k0+guard] in a single-sided spectrum mask.
% Inputs k0 are 0-based; MATLAB indices are 1-based.
    lo = max(0, k0-guard);
    hi = min(Knyq, k0+guard);
    mask((lo+1):(hi+1)) = true;
end