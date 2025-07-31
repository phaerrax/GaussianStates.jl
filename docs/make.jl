using Documenter, DocumenterCitations
using GaussianStates

push!(LOAD_PATH, "../src/")
# This line is needed because GaussianStates is not accessible through Julia's LOAD_PATH.

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style=:numeric)

makedocs(;
    modules=[GaussianStates],
    sitename="GaussianStates",
    repo=Remotes.GitHub("phaerrax", "GaussianStates.jl"),
    checkdocs=:exported,
    authors="Davide Ferracin",
    pages=["Home" => "index.md", "Reference" => "reference.md"],
    plugins=[bib],
    format=Documenter.HTML(;
        mathengine=Documenter.MathJax(
            Dict(
                :TeX => Dict(
                    :Macros => Dict(
                        :ket => [raw"\lvert #1 \rangle", 1],
                        :bra => [raw"\langle #1 \rvert", 1],
                        :transpose => [raw"#1^{\mathrm{T}}", 1],
                        :conj => [raw"\overline{#1}", 1],
                        :sympmat => [raw"\varOmega"],
                        :adj => [raw"#1^\dagger", 1],
                        :real => [raw"\operatorname{Re}"],
                        :imag => [raw"\operatorname{Im}"],
                        :N => [raw"\mathbb{N}"],
                        :C => [raw"\mathbb{C}"],
                        :R => [raw"\mathbb{R}"],
                        :dd => [raw"\mathrm{d}"],
                        :det => [raw"\operatorname{det}"],
                        :opt => [raw"_{\mathrm{opt}}"],
                        :tr => [raw"\operatorname{tr}"],
                    ),
                ),
            ),
        ),
    ),
)

# Automatically deploy documentation to gh-pages.
deploydocs(; repo="github.com/phaerrax/GaussianStates.jl.git")
