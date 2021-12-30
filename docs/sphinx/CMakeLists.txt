# The folder containing the bindings Python modules
set(MODULE_PATH "${PROJECT_SOURCE_DIR}")

find_package(SphinxApidoc REQUIRED)
add_custom_target(apidoc)

add_custom_command(
    TARGET apidoc POST_BUILD
    COMMAND
    ${SPHINX_APIDOC_EXECUTABLE} --force
    --no-toc --module-first --maxdepth 4
    -t ${CMAKE_CURRENT_SOURCE_DIR}/_templates
    -o ${CMAKE_CURRENT_SOURCE_DIR}/apidoc/gym-os2r
    ${MODULE_PATH}
    COMMENT "Building <gym-os2r> apidoc")

find_package(Sphinx REQUIRED)
find_package(SphinxMultiVersion REQUIRED)

set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/html)
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})

# Extend the Python path including before generating the website
set(SPHINX_CMD export PYTHONPATH="${MODULE_PATH}/gym_os2r" &&)

# Sphinx build command
message(STATUS "SPHNIX MV EXEC: ${SPHINX_MULTIVERSION_EXECUTABLE}")
list(APPEND SPHINX_CMD
    ${SPHINX_MULTIVERSION_EXECUTABLE}
    ${SPHINX_SOURCE} ${SPHINX_BUILD})

# Generate the website
add_custom_target(sphinx ALL
    COMMAND ${SPHINX_CMD}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating documentation with Sphinx")

# Disable GitHub pages autogeneration
add_custom_command(
    TARGET sphinx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E touch "${SPHINX_BUILD}/.nojekyll"
    COMMENT "Disabling Jekyll in html folder")

# Handle redirect
add_custom_command(
    TARGET sphinx POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/index.html" "${SPHINX_BUILD}"
    COMMENT "Copying html redirect to html folder")
