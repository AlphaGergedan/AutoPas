#pragma once

/**
 * Includes helpers for writing Vtk files, to be used in application examples (e.g. mdflexible)
 * FIXME: Should we disable tabs at the beginning? Maybe wrap into class?
 * FIXME: n_components is zero for 'types' in Particles file, why?
 */
namespace autopas::utils::Vtk {

/// Type definitions

enum class GridType {
  Unstructured,
  ParallelUnstructured,
};

enum class DataType {
  // Add more when needed, missing types: Int8, Int16, and unsigned integers e.g. UInt8
  Int32,
  Int64,
  Float32,
  Float64,
};

enum class Section {
  PointData,
  ParallelPointData,
  CellData,
  ParallelCellData,
  Points,
  ParallelPoints,
  Cells,
  ParallelCells,
  Piece,
};

constexpr std::string_view toString(GridType t) {
  switch (t) {
    case GridType::Unstructured:
      return "UnstructuredGrid";
    case GridType::ParallelUnstructured:
      return "PUnstructuredGrid";
  }
  return "";
}

constexpr std::string_view toString(DataType d) {
  switch (d) {
    case DataType::Int32:
      return "Int32";
    case DataType::Int64:
      return "Int64";
    case DataType::Float32:
      return "Float32";
    case DataType::Float64:
      return "Float64";
  }
  return "";
}

constexpr std::string_view toString(Section s) {
  switch (s) {
    case Section::PointData:
      return "PointData";
    case Section::ParallelPointData:
      return "PPointData";
    case Section::CellData:
      return "CellData";
    case Section::ParallelCellData:
      return "PCellData";
    case Section::Points:
      return "Points";
    case Section::ParallelPoints:
      return "PPoints";
    case Section::Cells:
      return "Cells";
    case Section::ParallelCells:
      return "PCells";
    case Section::Piece:
      return "Piece";
  }
  return "";
}

/// Header and footer

inline void beginFile(std::ostream &os, GridType gridType) {
  os << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\" ?>\n";
  os << "<VTKFile byte_order=\"LittleEndian\" type=\"" << toString(gridType) << "\" version=\"0.1\">\n";
  os << "  <" << toString(gridType);
  if (gridType == GridType::ParallelUnstructured) {
    os << " GhostLevel=\"0\"";
  }
  os << ">\n";
}

inline void endFile(std::ostream &os, GridType gridType) {
  os << "  </" << toString(gridType) << ">\n";
  os << "</VTKFile>\n";
}

// Sections

inline void addEmptySection(std::ostream &os, Section section) {
  os << "    <" << toString(section) << "/>\n";
}

inline void beginSection(std::ostream &os, Section section) {
  os << "    <" << toString(section) << ">\n";
}

inline void endSection(std::ostream &os, Section section) {
  os << "    </" << toString(section) << ">\n";
}

// Bulk data related helpers

inline void addParallelDataArray(
    std::ostream &os, DataType dataType, std::string_view name,
    int numberOfComponents, std::string_view format) {
  os << "      <PDataArray ";
  os << "type=\"" << toString(dataType) << "\" ";
  os << "Name=\"" << name << "\" ";
  os << "NumberOfComponents=\"" << numberOfComponents << "\" ";
  os << "format=\"" << format << "\"/>\n";
}

inline void addParallelPiece(std::ostream &os, std::string_view source) {
  os << "    <Piece ";
  os << "Source=\"" << source << "\"/>\n";
}

inline void beginPiece(std::ostream &os, long numCells, long numPoints) {
  os << "    <Piece ";
  os << "NumberOfCells=\"" << numCells << "\" ";
  os << "NumberOfPoints=\"" << numPoints << "\">\n";
}

}   // namespace autopas::utils::Vtk
