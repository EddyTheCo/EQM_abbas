cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
include(local_conf.cmake OPTIONAL)
project(EQM_module VERSION 0.1 DESCRIPTION "a Easy Quantum model ML-torch module for Fisher Matrix calculations" LANGUAGES CXX)

find_package(Torch REQUIRED)

include(FetchContent)
add_library(EQM_abbas include/custom-modules/eqm_abbas.hpp eqm_abbas.cpp)

if(USE_YAML)
	target_compile_definitions(EQM_abbas PUBLIC USE_YAML)
	if(NOT TARGET yaml-cpp)
		find_package(YAML-CPP CONFIG QUIET)
		if(NOT YAML-CPP_FOUND)
			message(STATUS "YAML-CPP was not found. Downloading")

			FetchContent_Declare(
				yaml-cpp
				GIT_REPOSITORY git@github.com:jbeder/yaml-cpp.git
				GIT_TAG yaml-cpp-0.7.0
				)
			FetchContent_MakeAvailable(yaml-cpp)
		else()
			message(STATUS "YAML FOUND in ${YAML-CPP_DIR}")
		endif(NOT YAML-CPP_FOUND)
	endif(NOT TARGET yaml-cpp)
	install(TARGETS yaml-cpp EXPORT EQM_abbasTargets DESTINATION lib)
	target_link_libraries(EQM_abbas PRIVATE  yaml-cpp)
endif(USE_YAML)

find_package(TBB CONFIG REQUIRED)
find_package(oneDPL CONFIG REQUIRED)

message(STATUS "Downloading qulacs")
FetchContent_Declare(
	qulacs
	GIT_REPOSITORY git@github.com:EddyTheCo/qulacs.git
	GIT_TAG main
	)
FetchContent_MakeAvailable(qulacs)
target_link_libraries(EQM_abbas PRIVATE  cppsim)

target_link_libraries(EQM_abbas  PUBLIC "${TORCH_LIBRARIES}")


target_link_libraries(EQM_abbas PRIVATE  ${TBB_IMPORTED_TARGETS} oneDPL)

target_include_directories(EQM_abbas PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
	"$<INSTALL_INTERFACE:include>")

install(TARGETS EQM_abbas EXPORT EQM_abbasTargets DESTINATION lib)
install(DIRECTORY include/ DESTINATION include/)

install(EXPORT EQM_abbasTargets
	FILE EQM_abbasTargets.cmake
	DESTINATION lib/cmake/EQM_abbas
	)
include(CMakePackageConfigHelpers)
# generate the config file that is includes the exports
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/Config.cmake.in
	"${CMAKE_CURRENT_BINARY_DIR}/EQM_abbasConfig.cmake"
	INSTALL_DESTINATION "lib/cmake/EQM_abbas"
	NO_SET_AND_CHECK_MACRO
	NO_CHECK_REQUIRED_COMPONENTS_MACRO
	)
write_basic_package_version_file(
	"${CMAKE_CURRENT_BINARY_DIR}/EQM_abbasConfigVersion.cmake"
	VERSION "0.1.1.0"
	COMPATIBILITY AnyNewerVersion
	)
install(FILES
	${CMAKE_CURRENT_BINARY_DIR}/EQM_abbasConfig.cmake
	${CMAKE_CURRENT_BINARY_DIR}/EQM_abbasConfigVersion.cmake
	DESTINATION lib/cmake/EQM_abbas
	)
export(EXPORT EQM_abbasTargets
	FILE "${CMAKE_CURRENT_BINARY_DIR}/EQM_abbasTargets.cmake"
	)
