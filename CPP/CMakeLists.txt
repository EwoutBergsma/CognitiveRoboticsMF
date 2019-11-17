cmake_minimum_required( VERSION 2.8 )

# Create Project
project( solution )
add_executable( project main.cpp )

# Set StartUp Project (Option)
# (This setting is able to enable by using CMake 3.6.0 RC1 or later.)
set_property( DIRECTORY PROPERTY VS_STARTUP_PROJECT "project" )

# Find Packages
find_package( PCL 1.8 REQUIRED )

if( PCL_FOUND )
  # [C/C++]>[General]>[Additional Include Directories]
  # OLD: include_directories( ${PCL_INCLUDE_DIRS} )
  include_directories( ${PCL_INCLUDE_DIRS} ${PROJECT_SOURCE_DIR}/include)

  # [C/C++]>[Preprocessor]>[Preprocessor Definitions]
  add_definitions( ${PCL_DEFINITIONS} )

  # For Use Not PreCompiled Features
  #add_definitions( -DPCL_NO_PRECOMPILE )

  # [Linker]>[General]>[Additional Library Directories]
  link_directories( ${PCL_LIBRARY_DIRS} )

  add_library (good good.cpp)
  SET_TARGET_PROPERTIES(good PROPERTIES LINK_FLAGS -L${PCL_LIBRARY_DIRS})
  target_link_libraries(good ${PCL_LIBRARIES})
  target_link_libraries(good boost_system)

  # [Linker]>[Input]>[Additional Dependencies]
  # OLD: target_link_libraries( project ${PCL_LIBRARIES} )
  # OLDER: target_link_libraries( project -lflann -lflann_cpp)
  target_link_libraries (project ${PCL_LIBRARIES} )
  target_link_libraries(project boost_system)
  target_link_libraries(project good)
endif()