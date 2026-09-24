-- -- Define a custom option
-- option("SIMPLE_ADD_FIELD")
--     set_default("bn254_fr::Element") -- Default value: TEMPLATE_A.
--     set_showmenu(true) -- Show this option in `xmake f --help`.
--     set_description("Choose a field for simple add")

-- target("simple_add")
--     set_kind("shared")
--     set_targetdir(os.projectdir().."/lib")
--     set_optimize("fastest")

--     if has_config("SIMPLE_ADD_FIELD") then
--         add_defines("FIELD="..get_config("SIMPLE_ADD_FIELD"))
--         set_basename("simple_add" .. "_" .. get_config("SIMPLE_ADD_FIELD"))
--     end

--     add_files("src/add.cu")

fields = {
    "bn254_fr",
    "bls12381_fr",
}

for _, field in ipairs(fields) do
    local target_name = "simple_add_" .. field
    target(target_name)
        set_kind("shared")
        add_cugencodes("native")
        set_targetdir(os.projectdir().."/lib")
        set_optimize("fastest")
        add_defines("FIELD="..field.."::Element")
        add_files("src/add.cu")
        set_languages("c++17")
end


