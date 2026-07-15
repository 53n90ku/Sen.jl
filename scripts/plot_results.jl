function read_tsv(path::AbstractString)
    lines=readlines(path)
    isempty(lines)&&error("result file is empty")
    headers=split(first(lines),'\t')
    return[Dict(headers[index]=>values[index] for index in eachindex(headers)) for values in (split(line,'\t') for line in lines[2:end])]
end

function number(row,key)
    return parse(Float64,row[key])
end

function escape_xml(value)
    return replace(string(value),'&'=>"&amp;",'<'=>"&lt;",'>'=>"&gt;",'"'=>"&quot;")
end

function format_number(value::Float64)
    value>=1000&&return string(round(Int,value))
    value>=100&&return string(round(value;digits=1,))
    value>=10&&return string(round(value;digits=2,))
    value>=1&&return string(round(value;digits=3,))
    return string(round(value;digits=4,))
end

function write_line_plot(path::AbstractString,title::String,ylabel::String,xvalues::Vector{Float64},series::AbstractVector{<:NamedTuple};log_y::Bool=false,)
    width=960
    height=600
    left=92
    right=230
    top=72
    bottom=76
    plot_width=width-left-right
    plot_height=height-top-bottom
    all_values=Float64[value for item in series for value in item.values]
    isempty(all_values)&&error("plot has no values")
    log_y&&any(value->value<=0,all_values)&&error("log plot values must be positive")
    transformed=log_y ? log10.(all_values) : all_values
    yminimum=minimum(transformed)
    ymaximum=maximum(transformed)

    if yminimum==ymaximum
        yminimum-=0.5
        ymaximum+=0.5
    else
        padding=(ymaximum-yminimum)*0.08
        yminimum-=padding
        ymaximum+=padding
    end

    xposition(index)=length(xvalues)==1 ? left+plot_width/2 : left+(index-1)*plot_width/(length(xvalues)-1)
    yposition(value)=begin
        transformed_value=log_y ? log10(value) : value
        top+plot_height-(transformed_value-yminimum)/(ymaximum-yminimum)*plot_height
    end
    mkpath(dirname(path))

    open(path,"w") do io
        println(io,"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"$(width)\" height=\"$(height)\" viewBox=\"0 0 $(width) $(height)\">")
        println(io,"<rect width=\"100%\" height=\"100%\" fill=\"#0b1020\"/>")
        println(io,"<text x=\"$(left)\" y=\"36\" fill=\"#f8fafc\" font-family=\"system-ui,sans-serif\" font-size=\"24\" font-weight=\"700\">$(escape_xml(title))</text>")

        for tick in 0:5
            transformed_value=yminimum+(ymaximum-yminimum)*tick/5
            value=log_y ? 10.0^transformed_value : transformed_value
            y=top+plot_height-tick*plot_height/5
            println(io,"<line x1=\"$(left)\" y1=\"$(y)\" x2=\"$(left+plot_width)\" y2=\"$(y)\" stroke=\"#26324d\" stroke-width=\"1\"/>")
            println(io,"<text x=\"$(left-12)\" y=\"$(y+5)\" text-anchor=\"end\" fill=\"#94a3b8\" font-family=\"ui-monospace,monospace\" font-size=\"13\">$(format_number(value))</text>")
        end

        for(index,value) in enumerate(xvalues)
            x=xposition(index)
            println(io,"<line x1=\"$(x)\" y1=\"$(top)\" x2=\"$(x)\" y2=\"$(top+plot_height)\" stroke=\"#18233a\" stroke-width=\"1\"/>")
            println(io,"<text x=\"$(x)\" y=\"$(top+plot_height+30)\" text-anchor=\"middle\" fill=\"#cbd5e1\" font-family=\"system-ui,sans-serif\" font-size=\"14\">$(format_number(value*100))%</text>")
        end

        println(io,"<line x1=\"$(left)\" y1=\"$(top+plot_height)\" x2=\"$(left+plot_width)\" y2=\"$(top+plot_height)\" stroke=\"#64748b\" stroke-width=\"2\"/>")
        println(io,"<line x1=\"$(left)\" y1=\"$(top)\" x2=\"$(left)\" y2=\"$(top+plot_height)\" stroke=\"#64748b\" stroke-width=\"2\"/>")

        for item in series
            points=join(("$(xposition(index)),$(yposition(value))" for(index,value) in enumerate(item.values)),' ')
            println(io,"<polyline points=\"$(points)\" fill=\"none\" stroke=\"$(item.color)\" stroke-width=\"3\" stroke-linejoin=\"round\" stroke-linecap=\"round\"/>")

            for(index,value) in enumerate(item.values)
                println(io,"<circle cx=\"$(xposition(index))\" cy=\"$(yposition(value))\" r=\"5\" fill=\"$(item.color)\" stroke=\"#0b1020\" stroke-width=\"2\"/>")
            end
        end

        legend_x=left+plot_width+28
        for(index,item) in enumerate(series)
            y=top+index*30
            println(io,"<line x1=\"$(legend_x)\" y1=\"$(y)\" x2=\"$(legend_x+28)\" y2=\"$(y)\" stroke=\"$(item.color)\" stroke-width=\"4\"/>")
            println(io,"<text x=\"$(legend_x+38)\" y=\"$(y+5)\" fill=\"#e2e8f0\" font-family=\"system-ui,sans-serif\" font-size=\"14\">$(escape_xml(item.label))</text>")
        end

        println(io,"<text x=\"$(left+plot_width/2)\" y=\"$(height-20)\" text-anchor=\"middle\" fill=\"#cbd5e1\" font-family=\"system-ui,sans-serif\" font-size=\"15\">Filter selectivity</text>")
        println(io,"<text x=\"24\" y=\"$(top+plot_height/2)\" text-anchor=\"middle\" fill=\"#cbd5e1\" font-family=\"system-ui,sans-serif\" font-size=\"15\" transform=\"rotate(-90 24 $(top+plot_height/2))\">$(escape_xml(ylabel))</text>")
        println(io,"</svg>")
    end

    return String(path)
end

function method_series(rows,key::String,methods)
    colors=Dict(
        "exact"=>"#94a3b8",
        "ivf_prefilter"=>"#60a5fa",
        "ivf_postfilter"=>"#f97316",
        "filter_aware"=>"#22c55e",
        "filter_aware_bound"=>"#a78bfa",
        "auto"=>"#facc15",
    )
    labels=Dict(
        "exact"=>"Exact filtered",
        "ivf_prefilter"=>"IVF prefilter",
        "ivf_postfilter"=>"IVF postfilter",
        "filter_aware"=>"Filter-aware IVF",
        "filter_aware_bound"=>"Bounded filter-aware",
        "auto"=>"AutoPlanner",
    )

    return[(label=labels[method],color=colors[method],values=[number(row,key) for row in sort([row for row in rows if row["method"]==method];by=row->number(row,"selectivity"),)],) for method in methods]
end

function main()
    result_path=isempty(ARGS) ? joinpath(@__DIR__,"..","results","final-100k") : abspath(first(ARGS))
    aggregate=read_tsv(joinpath(result_path,"aggregate.tsv"))
    claims=read_tsv(joinpath(result_path,"claims.tsv"))
    xvalues=sort!(unique(number(row,"selectivity") for row in aggregate))
    plot_path=joinpath(result_path,"plots")
    methods=["exact","ivf_prefilter","ivf_postfilter","filter_aware","filter_aware_bound","auto"]

    latency=write_line_plot(joinpath(plot_path,"p95_latency.svg"),"p95 latency at Recall@10 >= 0.95","p95 latency (ms, log scale)",xvalues,method_series(aggregate,"p95_median_ms",methods);log_y=true,)
    candidates=write_line_plot(joinpath(plot_path,"candidates_scored.svg"),"Average vectors scored","Candidates scored (log scale)",xvalues,method_series(aggregate,"candidates_scored_mean",methods);log_y=true,)
    speedup_series=[(label="Filter-aware vs postfilter",color="#22c55e",values=[number(row,"speedup_p95_median") for row in sort(claims;by=row->number(row,"selectivity"),)],)]
    speedup=write_line_plot(joinpath(plot_path,"p95_speedup.svg"),"p95 speedup over calibrated IVF postfilter","Speedup (x)",xvalues,speedup_series;log_y=false,)

    println("latency=$(latency)")
    println("candidates=$(candidates)")
    println("speedup=$(speedup)")
end

main()
