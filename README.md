# Matrix Helper Functions 

A comprehensive, production-ready Python utility module leveraging **NumPy** to perform standard and advanced matrix operations. This repository extends the core functionality inspired by the CPL104 curriculum, offering 15 robust functions equipped with error handling, shape verification, and precision tolerance.

## 🚀 Features

This library is divided into core operations and advanced helper utilities:

### 🔹 Core Functions (Original CPL104 Set)
* **Matrix Addition:** Computes the element-wise sum of two matrices ($A + B$).
* **Determinant:** Calculates the determinant of any square matrix.
* **Transpose:** Flips a matrix over its diagonal ($A^T$).
* **Symmetry Check:** Verifies if a matrix is symmetric ($A = A^T$) within a specified tolerance.
* **Matrix Multiplication:** Performs standard matrix dot products ($A \times B$) with inner-dimension validation.

### 🔸 Extended Utility Functions (10 New Additions)
* **Matrix Power:** Raises square matrices to positive/negative integer powers ($A^n$).
* **Matrix Inverse:** Computes $A^{-1}$ for non-singular matrices with safety checks.
* **Matrix Rank:** Detects the number of linearly independent rows/columns.
* **Matrix Trace:** Computes the sum of the main diagonal elements.
* **Identity Check:** Checks if a matrix matches the identity matrix ($I$).
* **Orthogonal Check:** Validates if a matrix's transpose is equal to its inverse ($A \cdot A^T = I$).
* **Diagonal Check:** Confirms if all off-diagonal entries are zero.
* **Matrix Norm:** Calculates various norms (Frobenius, Spectral, Max Absolute Row/Col).
* **Linear System Solver:** Solves $Ax = b$ efficiently using LU decomposition.
* **Kronecker Product:** Computes the tensor product ($A \otimes B$) generating high-dimensional block matrices.

---

## 📦 Prerequisites & Installation

Ensure you have Python 3.8+ and NumPy installed.

```bash
pip install numpy
