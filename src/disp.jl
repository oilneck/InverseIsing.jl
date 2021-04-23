struct TextPrinter
end

function dprint(dict::AbstractDict; reorder::Bool=false)
    if reorder
        dict = sort(dict, by = x -> x[1])
    end
    for key in keys(dict)
        print(key, " => ", dict[key], "\n")
    end
end
