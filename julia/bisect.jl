# Bisection algorithms.
# Taken from https://docs.python.org/3/library/bisect.html


function insort_right(a, x, lo=1, hi=nothing)
    #=Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the right of the rightmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    =#
    lo = bisect_right(a, x, lo, hi)
    insert!(a, lo, x)
end


function insort_left(a, x, lo=1, hi=nothing)
    #=Insert item x in list a, and keep it sorted assuming a is sorted.

    If x is already in a, insert it to the left of the leftmost x.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    =#
    lo = bisect_left(a, x, lo, hi)
    insert!(a, lo, x)
end


function bisect_left(a, x, lo=1, hi=nothing)
    #=Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e < x, and all e in
    a[i:] have e >= x.  So if x already appears in the list, a.insert(x) will
    insert just before the leftmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    =#
    if hi === nothing
        hi = length(a)+1
    end
    while lo < hi
        mid = (lo+hi)÷2
        if a[mid] < x
            lo = mid+1
        else
            hi = mid
        end
    end
    return lo
end


function bisect_right(a, x, lo=1, hi=nothing)
    #=Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    =#
    if hi === nothing
        hi = length(a)+1
    end
    while lo < hi
        mid = (lo+hi)÷2
        if x < a[mid]
            hi = mid
        else
            lo = mid + 1
        end
    end
    return lo
end