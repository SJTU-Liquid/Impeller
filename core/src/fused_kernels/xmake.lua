option("FUSED_FIELD")
    set_default("bn254_fr") -- Default value: TEMPLATE_A.
    set_showmenu(true) -- Show this option in `xmake f --help`.
    set_description("Choose a field for ntt") -- Set the option description.

target("fused_kernels")
    set_kind("shared")
    set_targetdir(os.projectdir().."/lib")
    add_files("src/*.cu")

    if has_config("FUSED_FIELD") then
        add_defines("FUSED_FIELD="..get_config("FUSED_FIELD").."::Element")
    end

    add_cugencodes("native")
    set_optimize("fastest")
    set_languages("c++17")

    -- This hook runs after all source files are prepared and can modify each file’s compiler options.
    before_build_files(function (target, sourcebatch, opt)
        for _, sourcefile in ipairs(sourcebatch.sourcefiles) do
            -- Extract the filename without its path.
            local filename = path.filename(sourcefile)
            -- Match trailing digits, such as `64` in `mykernel_64.cu`.
            local regcount = filename:match("(%d+)%.[^%.]+$")
            if regcount then
                local flag = "--maxrregcount=" .. regcount
                -- print("Applying to " .. filename .. ": " .. flag)
                -- Add this option to the source file.
                -- 256 is the biggest value for maxrregcount, so we don't need to set it
                if tonumber(regcount) < 256 then
                    target:fileconfig_set(sourcefile, {
                        cuflags = flag
                    })
                end
            end
        end
    end)
