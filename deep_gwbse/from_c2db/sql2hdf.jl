# Copying information the SQLite C2DB database
# nto an HDF5 file
# Just download `database_name` to this folder 
# and then run this script in this folder.

using SQLite, SQLite.DBInterface
using DataFrames
using HDF5

#################################
#region Configuration

database_name = "c2db-2022-11-30.db"
hdf_name      = "c2db-2022-11-30.h5"

println("Reading database :    $database_name")
println("Writing to HDF5  :    $hdf_name")
println()

#endregion 
#################################

#################################
#region Initialization: 
# obtain the list of materials, 
# and create the HDF5 file

db = SQLite.DB(database_name)
material_list = DBInterface.execute(db, 
    "select id, unique_id from systems") |> DataFrame
n_material = size(material_list)[1]

h5_fid = h5open(hdf_name, "w")

h5_fid["id"]        = material_list[:, "id"]
h5_fid["unique_id"] = material_list[:, "unique_id"]

println("HDF5 file created/overwritten.")

#endregion
#################################

#################################
#region Copy atomic information  

systems = DBInterface.execute(db, 
    "select numbers, positions, cell from systems") |> DataFrame
println("Database loading finished.")
println()

#region Atomic species

atomic_numbers = map(systems[:, "numbers"]) do atom_dataset
    reinterpret(Int32, atom_dataset)
end

# Due to the fact that each material has a different number of atoms, 
# it's impossible to define a multidimensional array in the output HDF5 file.
# We instead reshape `atom_numbers` into a one-dimensional array,
# and specify the number of atoms in each system, 
# so that 
h5_fid["numbers_of_atoms"] = map(length, atomic_numbers)
h5_fid["atoms"]            = vcat(atomic_numbers...)

println("Finish copying atomic species.")

#endregion

#region Cell  

# Format inferred from the pattern of data: 
# h5_fid["cell"][nth_coordinate, nth_basis_vector, material_index]
h5_fid["cell"]  = zeros(3, 3, n_material)
map(enumerate(systems[:, "cell"])) do (material_idx, cell_dataset) 
    cell_array = reshape(
        reinterpret(Float64, cell_dataset),
        (3, 3)
    )  |> Matrix
    
    h5_fid["cell"][:, :, material_idx] = cell_array
end

println("Finish copying lattice structure.")

#endregion

#region Positions of atoms 

# Format: 
# h5_fid["positions"][nth_coordinate, idx_atom], 
# and h5_fid["positions"][1, :] 
# contains x coordinates of the first, second, third, ... atom of the first material,
# and then the second material, and then the third material, etc.
h5_fid["positions"] = zeros(3, sum(map(length, atomic_numbers)))
let atom_count = 0
    map(enumerate(systems[:, "positions"])) do (material_idx, positions_dataset) 
        n_atom = length(atomic_numbers[material_idx])
        positions = reshape(
            reinterpret(Float64, positions_dataset), 
            (3, n_atom)) |> Matrix 
        for idx_atom in 1 : n_atom
            atom_count += 1
            h5_fid["positions"][:, atom_count] = positions[:, idx_atom]
        end
    end
end

println("Finish copying atomic positions.")

#endregion

#endregion
#################################

close(h5_fid)
println()
println("All done!")