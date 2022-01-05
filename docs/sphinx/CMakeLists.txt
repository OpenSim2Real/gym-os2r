# The folder containing the bindings Python modules
set(MODULE_PATH ${PROJECT_SOURCE_DIR})

find_package(SphinxApidoc REQUIRED)
set(SPHINX_BUILD ${CMAKE_BINARY_DIR}/html)
set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})

add_custom_target( apidoc ALL
    COMMAND
    ${SPHINX_APIDOC_EXECUTABLE} --force
    --no-toc --module-first --maxdepth 4
    -t ${CMAKE_CURRENT_SOURCE_DIR}/_templates
    -o ${SPHINX_SOURCE}/gym-os2r_docs/python/gym-os2r
    ${MODULE_PATH}/gym_os2r
    COMMENT "Building <gym-os2r> apidoc")

find_package(Sphinx REQUIRED)
find_package(SphinxMultiVersion REQUIRED)

# =============
# Create built package for overlord webpage in /html
# =============
add_custom_target(source_rst_copy ALL
    DEPENDS apidoc
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${SPHINX_SOURCE}/gym-os2r_docs" "${SPHINX_BUILD}/sphinx_build/gym-os2r_docs"
    COMMENT "copy source doc build files."
  )

add_custom_target(index_copy ALL
    DEPENDS apidoc
    COMMAND ${CMAKE_COMMAND} -E copy "${SPHINX_SOURCE}/index.rst" "${SPHINX_BUILD}/sphinx_build/gym-os2r_index.rst"
    COMMENT "copy index rst."
  )

# =============
# Compile Docs for Repo.
# =============

IF(NOT BUILD_DOCS_SUPERBUILD)
    # Extend the Python path including before generating the website
    message(STATUS "Python paths for gym-os2r build: ${PYTHONPATHS}")
    set(SPHINX_CMD ${CMAKE_COMMAND} -E env PYTHONPATH="${PYTHONPATHS}")

    # Sphinx build command
    message(STATUS "SPHNIX MV EXEC: ${SPHINX_MULTIVERSION_EXECUTABLE}")
    list(APPEND SPHINX_CMD ${SPHINX_MULTIVERSION_EXECUTABLE} ${SPHINX_SOURCE} ${SPHINX_BUILD})

    # Generate the website
    add_custom_target(sphinx ALL
      COMMAND ${SPHINX_CMD}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
      COMMENT "Generating documentation with Sphinx"
    )

    # Disable GitHub pages autogeneration
    add_custom_command(
      TARGET sphinx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E touch "${SPHINX_BUILD}/.nojekyll"
      COMMENT "Disabling Jekyll in html folder"
    )

    # Handle redirect
    add_custom_command(
      TARGET sphinx POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/index.html" "${SPHINX_BUILD}"
      COMMENT "Copying html redirect to html folder"
    )
ENDIF()
