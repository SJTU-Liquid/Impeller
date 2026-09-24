
option("NTT_FIELD")
    set_default("bn254_fr") -- Default value: TEMPLATE_A.
    set_showmenu(true) -- Show this option in `xmake f --help`.
    set_description("Choose a field for ntt") -- Set the option description.

target("ntt")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/ntt.cu")
    if has_config("NTT_FIELD") then
        add_defines("NTT_FIELD="..get_config("NTT_FIELD").."::Element")
    end
    add_cugencodes("native")
    set_optimize("fastest")
    set_languages("c++17")

target("test_ssip")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_ssip.cu")
    set_languages("c++17")

target("test_recompute")
    set_kind("binary")
    add_cugencodes("native")
    add_files("tests/test_recompute.cu")
    set_languages("c++17")
