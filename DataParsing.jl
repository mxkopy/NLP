using FileIO, DelimitedFiles, DataStructures, StringEncodings

prefix = pwd()

function parse_line( line::String; delim = ',', replacements=[ "\"" => "", "!" => "", "." => "" ] )

    stack::Queue{String} = Queue{String}()

    out = ""

    for replacement in replacements 

        line = replace(line, replacement)

    end

    for char in line

        if char == delim && length( stack ) < 6

            enqueue!( stack, out )
            out = ""

        else

            out = out * char

        end

    end

    return collect( stack )

end
