
# Mat

**一个支持自定义数值类型模板的 C++ 数值线性代数库：高性能、可套用符号模板、可选 AVX512。**

- 作者：xiangzi001
- 仓库：<https://github.com/xiangzi001/Mat>
- 当前版本：**v0.0，第 1 个正式发布版**
- 开发历程：从 2023 年开始持续开发，已写 3 年
- 详细说明：[Mat-release-必读.txt](./Mat-release-必读.txt)
- 使用本库时请注明出处：**Mat by xiangzi001, https://github.com/xiangzi001/Mat**

如果觉得好用，欢迎 Star、转发、推荐给同学/同事，或在论文、项目、博客中使用并注明出处。

---

## 亮点

Mat 是一个以 `Mat.h` 为核心的 C++ 数值线性代数库，重点面向：

- **double 下的小矩阵高性能计算**
- **单线程 double 下的大矩阵分块优化**
- **自定义数值类型模板**：分式、根式、`float128` 等
- **高精度方法**：Jacobi 的 `EVD_H` 和 `SVD`，GramSchmidt 的 `QR`
- **可选 AVX512 加速**
- **控制台友好的矩阵输入输出**

作者自测中，Mat 在 double 小矩阵上所有函数速度都很快；在单线程 double 大矩阵上，已完成大矩阵分块优化的函数也具备很强竞争力。测试数据可查看仓库中的 `Test-2000` 文件。

在 **单线程double** 前提下：
- **小矩阵**：作者测试中，所有函数速度都比市面上任何数值线性代数库都快。
- **大矩阵、单线程**：所有做过大矩阵分块优化的函数，速度都比市面上任何数值线性代数库都快，包括：
  - 乘法
  - PLU
  - Inv
  - LLT
  - LDLT
  - QR
  - EV_H
  - EVD_H(CDC)
- **EVD**：200 阶以内，作者测试比 OpenBLAS 快很多。
- **SVD**：72 阶以内，作者测试更快。
- 大矩阵测试数据见：`Test-2000`。

> 以下性能结论为作者在当前硬件、编译器和测试数据下的自测结果，欢迎复现与反馈。

---

## 使用和后续更新，见Mat-Release-必读.txt

## 参与与推广

如果 Mat 对你有帮助：

- 欢迎给仓库点一个 **Star**
- 欢迎转发给做数值线性代数、C++ 模板、高精度计算的朋友
- 欢迎提交 Issue 反馈问题或性能测试结果
- 欢迎 PR 贡献代码
- 使用、引用或二次分发时请注明出处：

```text
Mat — C++ numerical linear algebra library
Author: xiangzi001
Repository: https://github.com/xiangzi001/Mat
```

---

## License

Copyright (C) 2026 xiangzi001 <https://github.com/xiangzi001/Mat>

For details, see Mat-release-必读.txt.

This project is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

SPDX-License-Identifier: LGPL-3.0-or-later

This project is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
and the GNU General Public License along with this project.
If not, see <https://www.gnu.org/licenses/>.

Full license texts:
- COPYING: GNU General Public License version 3
- COPYING.LESSER: GNU Lesser General Public License version 3
