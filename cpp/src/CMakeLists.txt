file(GLOB DRIVER_SOURCES *.cpp)
foreach(src ${DRIVER_SOURCES})
get_filename_component(name ${src} NAME_WE)
add_executable(${name} ${src})
target_link_libraries(${name} PRIVATE caterpillar
mockturtle)
endforeach()