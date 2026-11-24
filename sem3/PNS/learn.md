# Linear Regression as Estimation

> [!info] Source
> MA 6.101 Probability and Statistics, `linear_regression.pdf`, slides 1–8 (up to and including *"A different noise model gives a different loss"*).

**The big idea:** fitting a line to data is not just geometry. It is **parameter estimation**. The true line has unknown parameters $\theta^* = (\beta_0^*, \beta_1^*)$, the data is random, and any line we compute from the data is an **estimator** with its own bias, variance and MSE, just like $\bar X$.

---

## 1. The Model (slide 2)

**Data:** pairs $D = \{(X_i, Y_i)\}_{i=1}^n$, observed as numbers $(x_i, y_i)$. For example, $x$ = hours studied and $y$ = marks.

**Model:**

$$
Y_i = \beta_0^* + \beta_1^* X_i + \varepsilon_i
$$

- $\beta_0^*$ (intercept) and $\beta_1^*$ (slope) are **unknown**. They are our $\theta^*$.
- $\varepsilon_1, \dots, \varepsilon_n$ is the **noise**: i.i.d., independent of the $X_i$, with

$$
E[\varepsilon_i] = 0, \qquad \text{Var}(\varepsilon_i) = \sigma^2
$$

**Consequence:**

$$
E[Y_i \mid X_i = x_i] = \beta_0^* + \beta_1^* x_i, \qquad \text{Var}(Y_i \mid X_i = x_i) = \sigma^2
$$

> [!tip] How to picture it
> At every $x_i$, $Y_i$ is a random variable with its own distribution (the blue curves in the slide), **centred on the true line**. The red data points you observe are **one random draw from each** of these distributions. The true line goes through the *conditional means*, not through the data points.

### Solved Example 1.1: Reading the model

> [!example] Problem
> Marks follow $Y = 20 + 7X + \varepsilon$ with $\varepsilon \sim N(0, 16)$, where $X$ = hours studied. A student studies 5 hours.
> (a) What are the expected marks? (b) What is the standard deviation of the marks? (c) What is the probability of scoring above 63?

**Solution.**

(a) $E[Y \mid X=5] = 20 + 7(5) = 55$.

(b) $\text{Var}(Y \mid X=5) = \sigma^2 = 16$, so the SD is $4$. The noise variance is the same at every $x$.

(c) $Y \mid X=5 \sim N(55, 16)$, so

$$
P(Y > 63) = P\!\left(Z > \frac{63 - 55}{4}\right) = P(Z > 2) \approx 0.0228
$$

---

## 2. Least Squares (slide 3)

For a candidate line $\beta_0 + \beta_1 x$, the **residual** of point $i$ is its vertical gap from the line:

$$
e_i = y_i - \beta_0 - \beta_1 x_i
$$

We want **all residuals small together**. Summing them does not work, because positive and negative residuals cancel. So we square them and add, which gives the **sum of squared errors**:

$$
J(\beta_0, \beta_1) = \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i)^2
$$

The **least squares estimates** $(\hat\beta_0, \hat\beta_1)$ are the values that **minimise** $J$.

### Solved Example 2.1: Comparing two candidate lines

> [!example] Problem
> Data: $(1,2), (2,3), (3,5), (4,6)$. Which line has the smaller $J$: $y = 1 + x$ or $y = 0.5 + 1.4x$?

**Solution.**

| $x_i$ | $y_i$ | Line A: $1+x$ | $e_i$ | $e_i^2$ | Line B: $0.5+1.4x$ | $e_i$ | $e_i^2$ |
|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 0 | 0 | 1.9 | 0.1 | 0.01 |
| 2 | 3 | 3 | 0 | 0 | 3.3 | −0.3 | 0.09 |
| 3 | 5 | 4 | 1 | 1 | 4.7 | 0.3 | 0.09 |
| 4 | 6 | 5 | 1 | 1 | 6.1 | −0.1 | 0.01 |

$J_A = 2$ and $J_B = 0.2$, so **line B is better**. As Example 4.1 shows, B is the least squares line, and no line can do better than $J = 0.2$.

---

## 3. Minimising $J$: the Intercept (slide 4)

**Notation:** the sample means are

$$
\bar x = \frac{1}{n}\sum_{i=1}^n x_i, \qquad \bar y = \frac{1}{n}\sum_{i=1}^n y_i
$$

**Useful fact:** deviations from the mean sum to zero.

$$
\sum_{i=1}^n (x_i - \bar x) = \sum_{i=1}^n x_i - n\bar x = 0, \qquad \sum_{i=1}^n (y_i - \bar y) = 0
$$

**Set both partial derivatives of $J$ to zero.** These are called the *normal equations*:

$$
\frac{\partial J}{\partial \beta_0} = -2\sum_{i=1}^n (y_i - \beta_0 - \beta_1 x_i) = 0
$$

$$
\frac{\partial J}{\partial \beta_1} = -2\sum_{i=1}^n x_i\,(y_i - \beta_0 - \beta_1 x_i) = 0
$$

Divide the first equation by $-2n$:

$$
\bar y - \beta_0 - \beta_1 \bar x = 0 \quad\Longrightarrow\quad \boxed{\beta_0 = \bar y - \beta_1 \bar x}
$$

> [!important] Consequence
> Rearranged, this reads $\bar y = \beta_0 + \beta_1 \bar x$. **The least squares line always passes through $(\bar x, \bar y)$.**

> [!note] Hidden meaning of the normal equations
> In terms of residuals, the two equations say
> $$\sum_i e_i = 0 \qquad \text{and} \qquad \sum_i x_i e_i = 0$$
> The fitted residuals always satisfy these two constraints. This matters again in Section 6.

---

## 4. Minimising $J$: the Slope (slide 5)

Substitute $\beta_0 = \bar y - \beta_1 \bar x$ into the second equation. Each term becomes $y_i - \bar y - \beta_1(x_i - \bar x)$, so

$$
\sum_{i=1}^n x_i c_i = 0, \qquad \text{where } c_i = (y_i - \bar y) - \beta_1(x_i - \bar x)
$$

**The trick:** the $c_i$ also sum to zero.

$$
\sum_i c_i = \sum_i (y_i - \bar y) - \beta_1 \sum_i (x_i - \bar x) = 0 - 0 = 0 \quad\Longrightarrow\quad \sum_i \bar x\, c_i = 0
$$

Subtracting this from $\sum_i x_i c_i = 0$ gives

$$
\sum_{i=1}^n (x_i - \bar x)\, c_i = 0 \quad\Longrightarrow\quad \sum_{i=1}^n (x_i - \bar x)(y_i - \bar y) = \beta_1 \sum_{i=1}^n (x_i - \bar x)^2
$$

Define

$$
S_{xx} = \sum_i (x_i - \bar x)^2, \qquad S_{xy} = \sum_i (x_i - \bar x)(y_i - \bar y)
$$

> [!success] Least squares estimates
> $$\hat\beta_1 = \frac{S_{xy}}{S_{xx}}, \qquad \hat\beta_0 = \bar y - \hat\beta_1 \bar x$$

> [!tip] Intuition
> $S_{xy}/n$ is the sample **covariance** of $x$ and $y$, and $S_{xx}/n$ is the sample **variance** of $x$. So the slope is $\hat\beta_1 = \widehat{\text{Cov}}(x,y) / \widehat{\text{Var}}(x)$. When $x$ and $y$ move together, the slope is positive.

> [!tip] Shortcut formulas (handy in exams)
> $$S_{xx} = \sum x_i^2 - n\bar x^2, \qquad S_{xy} = \sum x_i y_i - n\bar x \bar y$$

### Solved Example 4.1: A full LS fit

> [!example] Problem
> Find the least squares line for $(1,2), (2,3), (3,5), (4,6)$.

**Solution.** $\bar x = 10/4 = 2.5$ and $\bar y = 16/4 = 4$.

| $x_i$ | $y_i$ | $x_i - \bar x$ | $y_i - \bar y$ | $(x_i-\bar x)(y_i-\bar y)$ | $(x_i - \bar x)^2$ |
|---|---|---|---|---|---|
| 1 | 2 | −1.5 | −2 | 3 | 2.25 |
| 2 | 3 | −0.5 | −1 | 0.5 | 0.25 |
| 3 | 5 | 0.5 | 1 | 0.5 | 0.25 |
| 4 | 6 | 1.5 | 2 | 3 | 2.25 |
| **Sum** | | 0 | 0 | $S_{xy} = 7$ | $S_{xx} = 5$ |

$$
\hat\beta_1 = \frac{7}{5} = 1.4, \qquad \hat\beta_0 = 4 - 1.4 \times 2.5 = 0.5
$$

**Fitted line:** $\hat y = 0.5 + 1.4x$.

**Check with the shortcut formulas:** $\sum x_i^2 = 30$, so $S_{xx} = 30 - 4(6.25) = 5$ ✓. $\sum x_i y_i = 2+6+15+24 = 47$, so $S_{xy} = 47 - 4(2.5)(4) = 7$ ✓.

**Check the residual constraints:** the residuals are $0.1, -0.3, 0.3, -0.1$.
- $\sum e_i = 0$ ✓
- $\sum x_i e_i = 0.1 - 0.6 + 0.9 - 0.4 = 0$ ✓

### Solved Example 4.2: A second LS fit

> [!example] Problem
> Find the LS line for $x = 0,1,2,3,4$ and $y = 1,3,2,5,4$, then predict $y$ at $x = 6$.

**Solution.** $\bar x = 2$ and $\bar y = 15/5 = 3$.

| $x_i$ | $y_i$ | $x_i - \bar x$ | $y_i - \bar y$ | product | $(x_i-\bar x)^2$ |
|---|---|---|---|---|---|
| 0 | 1 | −2 | −2 | 4 | 4 |
| 1 | 3 | −1 | 0 | 0 | 1 |
| 2 | 2 | 0 | −1 | 0 | 0 |
| 3 | 5 | 1 | 2 | 2 | 1 |
| 4 | 4 | 2 | 1 | 2 | 4 |
| **Sum** | | | | $S_{xy}=8$ | $S_{xx}=10$ |

$$
\hat\beta_1 = 0.8, \qquad \hat\beta_0 = 3 - 0.8(2) = 1.4, \qquad \hat y = 1.4 + 0.8x
$$

**Prediction:** $\hat y(6) = 1.4 + 4.8 = 6.2$.

### Solved Example 4.3: Regression through the origin (exam-style variation)

> [!example] Problem
> Suppose the model has no intercept: $Y_i = \beta x_i + \varepsilon_i$. Find the least squares estimate $\hat\beta$.

**Solution.** Now $J(\beta) = \sum_i (y_i - \beta x_i)^2$. Set the derivative to zero:

$$
\frac{dJ}{d\beta} = -2\sum_i x_i (y_i - \beta x_i) = 0 \quad\Longrightarrow\quad \hat\beta = \frac{\sum_i x_i y_i}{\sum_i x_i^2}
$$

For the data in Example 4.1, $\hat\beta = 47/30 \approx 1.567$. This is different from 1.4, because the line is now forced through $(0,0)$ instead of $(\bar x, \bar y)$.

---

## 5. Least Squares is the MLE under Gaussian Noise (slide 6)

*Why squares? Why not absolute values, or fourth powers?* This section gives the answer.

**Assume** $\varepsilon_i \sim N(0, \sigma^2)$ i.i.d. Then $Y_1, \dots, Y_n$ are independent with

$$
Y_i \sim N(\beta_0^* + \beta_1^* x_i,\ \sigma^2)
$$

**Likelihood.** By independence, the joint density is a product:

$$
L(y_1,\dots,y_n;\ \beta_0, \beta_1) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left( -\frac{(y_i - \beta_0 - \beta_1 x_i)^2}{2\sigma^2} \right)
$$

**Log-likelihood.** The log turns the product into a sum, and the exponents add up to exactly $J$:

$$
\ln L(y_1,\dots,y_n;\ \beta_0, \beta_1) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} J(\beta_0, \beta_1)
$$

The first term does not depend on $(\beta_0, \beta_1)$, and $\frac{1}{2\sigma^2} > 0$. Therefore

$$
\text{maximising } \ln L \iff \text{minimising } J
$$

> [!success] Result
> $$(\hat\beta_0, \hat\beta_1)_{ML} = \arg\max_{\beta_0,\beta_1} \ln L = \arg\min_{\beta_0,\beta_1} J(\beta_0,\beta_1)$$
> $$\hat\beta_{1,ML} = \frac{S_{xy}}{S_{xx}}, \qquad \hat\beta_{0,ML} = \bar y - \hat\beta_{1,ML}\,\bar x$$
> **Under Gaussian noise, least squares = maximum likelihood.**

> [!tip] Why this works
> The Gaussian density has $(\text{residual})^2$ in its exponent. Taking logs exposes that square, so maximising the likelihood becomes minimising squared residuals.

### Solved Example 5.1: Comparing likelihoods of two lines

> [!example] Problem
> Using the data of Example 2.1 with known $\sigma^2 = 1$, by how much does the log-likelihood of line B ($0.5 + 1.4x$) exceed that of line A ($1 + x$)?

**Solution.** The constant $-\frac{n}{2}\ln(2\pi)$ cancels in the difference:

$$
\ln L_B - \ln L_A = -\frac{1}{2}(J_B - J_A) = -\frac{1}{2}(0.2 - 2) = 0.9
$$

So $L_B / L_A = e^{0.9} \approx 2.46$. Line B makes the observed data about 2.5 times more likely, and B is the line with the smaller $J$. This is the "max $\ln L$ = min $J$" result in numbers.

---

## 6. MLE of the Noise Variance (slide 7)

If $\sigma^2$ is **also unknown**, the log-likelihood is a function of three parameters:

$$
\ln L(y_1,\dots,y_n;\ \beta_0,\beta_1,\sigma^2) = -\frac{n}{2}\ln(2\pi\sigma^2) - \frac{1}{2\sigma^2} J(\beta_0,\beta_1)
$$

**Step 1.** For *every* fixed $\sigma^2$, the maximiser over $(\beta_0, \beta_1)$ is still $(\hat\beta_0, \hat\beta_1)$, because $\sigma^2$ only rescales $J$.

**Step 2.** Plug them in and maximise over $\sigma^2$, treating $\sigma^2$ as the variable:

$$
\frac{\partial \ln L}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{J(\hat\beta_0,\hat\beta_1)}{2\sigma^4} = 0
$$

Multiply through by $2\sigma^4$ to get $-n\sigma^2 + J = 0$, so:

> [!success] Result
> $$\hat\sigma^2_{ML} = \frac{J(\hat\beta_0,\hat\beta_1)}{n} = \frac{1}{n}\sum_{i=1}^n e_i^2, \qquad e_i = y_i - \hat\beta_0 - \hat\beta_1 x_i$$
> This is the average squared residual.

> [!warning] It is biased
> Like the sample variance with $\frac{1}{n}$, $\hat\sigma^2_{ML}$ is **biased** and tends to underestimate $\sigma^2$. The **unbiased** estimator is
> $$s^2 = \frac{1}{n-2}\sum_{i=1}^n e_i^2$$

> [!question] Why $n-2$?
> We estimated **two** parameters ($\beta_0, \beta_1$) from the data. That forces two constraints on the residuals: $\sum e_i = 0$ and $\sum x_i e_i = 0$. Only $n-2$ residuals are free to vary, so there are $n-2$ *degrees of freedom*.
> Compare the ordinary sample variance: it estimates **one** mean, has one constraint $\sum (x_i - \bar x) = 0$, and divides by $n-1$.

### Solved Example 6.1: Estimating $\sigma^2$

> [!example] Problem
> For the data in Example 4.1, find $\hat\sigma^2_{ML}$ and the unbiased estimate $s^2$.

**Solution.** The residuals are $0.1, -0.3, 0.3, -0.1$, so $J = 0.01 + 0.09 + 0.09 + 0.01 = 0.2$ and $n = 4$.

$$
\hat\sigma^2_{ML} = \frac{0.2}{4} = 0.05, \qquad s^2 = \frac{0.2}{4-2} = 0.1
$$

### Solved Example 6.2: Estimating $\sigma^2$ again

> [!example] Problem
> For the data in Example 4.2, find $\hat\sigma^2_{ML}$ and $s^2$.

**Solution.** The fitted values from $\hat y = 1.4 + 0.8x$ are $1.4, 2.2, 3.0, 3.8, 4.6$.

| $x_i$ | $y_i$ | $\hat y_i$ | $e_i$ | $e_i^2$ |
|---|---|---|---|---|
| 0 | 1 | 1.4 | −0.4 | 0.16 |
| 1 | 3 | 2.2 | 0.8 | 0.64 |
| 2 | 2 | 3.0 | −1.0 | 1.00 |
| 3 | 5 | 3.8 | 1.2 | 1.44 |
| 4 | 4 | 4.6 | −0.6 | 0.36 |
| **Sum** | | | 0 ✓ | $J = 3.6$ |

$$
\hat\sigma^2_{ML} = \frac{3.6}{5} = 0.72, \qquad s^2 = \frac{3.6}{3} = 1.2
$$

Check the second constraint: $\sum x_i e_i = 0 + 0.8 - 2.0 + 3.6 - 2.4 = 0$ ✓.

---

## 7. A Different Noise Model Gives a Different Loss (slide 8)

Suppose the noise is **Laplace** instead of Gaussian:

$$
f_\varepsilon(e) = \frac{1}{2b}\, e^{-|e|/b}
$$

Repeat the MLE steps from Section 5:

$$
\ln L = \sum_{i=1}^n \ln\!\left(\frac{1}{2b} e^{-|e_i|/b}\right) = -n\ln(2b) - \frac{1}{b}\sum_{i=1}^n |e_i|, \qquad e_i = y_i - \beta_0 - \beta_1 x_i
$$

So the MLE **minimises $\sum_i |e_i|$**. This method is called **least absolute deviations (LAD)**.

> [!important] The noise model decides the loss
> | Noise assumption | Log-density contains | Loss the MLE minimises | Name |
> |---|---|---|---|
> | Gaussian $N(0,\sigma^2)$ | $-e^2/(2\sigma^2)$ | $\sum e_i^2$ | Least squares (LS) |
> | Laplace$(b)$ | $-\lvert e\rvert/b$ | $\sum \lvert e_i\rvert$ | Least absolute deviations (LAD) |

> [!tip] Outliers: LS vs LAD
> - **LS:** squaring makes big residuals very expensive. A residual of 10 costs $100$. One outlier can **drag the LS line** towards itself (the blue line in the slide).
> - **LAD:** a residual of 10 costs only $10$. One outlier cannot pull the line much, so LAD is **robust** (the red line in the slide).
> - The Laplace distribution has **heavier tails** than the Gaussian. A model that expects occasional large errors does not overreact to them.

> [!tip] LS : LAD :: mean : median
> If you fit just a constant $c$ (no slope):
> - $\min_c \sum (y_i - c)^2 \Rightarrow c = \text{mean}$, which is sensitive to outliers.
> - $\min_c \sum |y_i - c| \Rightarrow c = \text{median}$, which is robust.
>
> LS and LAD are the regression versions of the mean and the median.

### Solved Example 7.1: Mean vs median as MLEs

> [!example] Problem
> Model $Y_i = \mu + \varepsilon_i$ with data $2, 3, 4, 5, 100$. Find the MLE of $\mu$ under (a) Gaussian noise and (b) Laplace noise.

**Solution.**

(a) Gaussian: minimise $\sum (y_i - \mu)^2$. Setting the derivative $-2\sum(y_i - \mu) = 0$ gives $\hat\mu = \bar y = 114/5 = 22.8$.

(b) Laplace: minimise $g(\mu) = \sum |y_i - \mu|$. The derivative (where it exists) is

$$
g'(\mu) = \#\{y_i < \mu\} - \#\{y_i > \mu\}
$$

It changes sign where half the points lie on each side, which is the **median**, so $\hat\mu = 4$.

The single outlier $100$ pulls the mean to $22.8$, a value larger than four of the five data points. The median ignores it.

### Solved Example 7.2: LS vs LAD lines with an outlier

> [!example] Problem
> Data: $(1,1), (2,2), (3,3), (4,4), (5,15)$. The last point is an outlier. Compare the LS line with the line $y = x$ under both losses.

**Solution.**

**LS line:** $\bar x = 3$, $\bar y = 25/5 = 5$.

| $x_i$ | $y_i$ | $x_i-\bar x$ | $y_i - \bar y$ | product | $(x_i-\bar x)^2$ |
|---|---|---|---|---|---|
| 1 | 1 | −2 | −4 | 8 | 4 |
| 2 | 2 | −1 | −3 | 3 | 1 |
| 3 | 3 | 0 | −2 | 0 | 0 |
| 4 | 4 | 1 | −1 | −1 | 1 |
| 5 | 15 | 2 | 10 | 20 | 4 |
| **Sum** | | | | $S_{xy}=30$ | $S_{xx}=10$ |

$$
\hat\beta_1 = 3, \qquad \hat\beta_0 = 5 - 3(3) = -4, \qquad \text{LS line: } y = 3x - 4
$$

The outlier has **tripled the slope**, even though four of the five points lie exactly on $y = x$.

**Compare the two losses for each line:**

| Line | Residuals $e_i$ | $\sum e_i^2$ (LS loss) | $\sum \lvert e_i\rvert$ (LAD loss) |
|---|---|---|---|
| $y = 3x - 4$ | $2, 0, -2, -4, 4$ | $\mathbf{40}$ | $12$ |
| $y = x$ | $0, 0, 0, 0, 10$ | $100$ | $\mathbf{10}$ |

- Under **squared loss**, $y = 3x-4$ wins, because squaring makes the outlier's residual of 10 cost 100.
- Under **absolute loss**, $y = x$ wins. It is in fact the LAD line here: it fits the four good points perfectly and pays only $10$ for the outlier.

So the Gaussian MLE (LS) is pulled by the outlier, while the Laplace MLE (LAD) is robust. This is the picture on slide 8.

> [!note] LAD has no closed form
> $\lvert e\rvert$ is not differentiable at 0, so there is no neat formula like $S_{xy}/S_{xx}$. In practice LAD is solved numerically, for example by linear programming. For exam problems, comparing $\sum |e_i|$ across candidate lines, as above, is usually enough.

### Solved Example 7.3: MLE of the Laplace scale $b$ (extension, not on the slides)

> [!example] Problem
> Under Laplace noise, find the MLE of $b$ once the line is fitted.

**Solution.** $\ln L = -n\ln(2b) - \frac{1}{b}\sum |e_i|$. Differentiate with respect to $b$:

$$
\frac{\partial \ln L}{\partial b} = -\frac{n}{b} + \frac{\sum |e_i|}{b^2} = 0 \quad\Longrightarrow\quad \hat b_{ML} = \frac{1}{n}\sum_{i=1}^n |e_i|
$$

This mirrors the Gaussian case, where $\hat\sigma^2_{ML} = \frac{1}{n}\sum e_i^2$. For Example 7.2 with $y = x$: $\hat b = 10/5 = 2$.

---

## Summary

| Concept | Formula / Fact |
|---|---|
| Model | $Y_i = \beta_0^* + \beta_1^* X_i + \varepsilon_i$, with $E[\varepsilon]=0$ and $\text{Var}(\varepsilon)=\sigma^2$ |
| Residual | $e_i = y_i - \beta_0 - \beta_1 x_i$ |
| LS objective | $J = \sum e_i^2$ |
| Slope | $\hat\beta_1 = S_{xy}/S_{xx}$ |
| Intercept | $\hat\beta_0 = \bar y - \hat\beta_1 \bar x$ (the line passes through $(\bar x,\bar y)$) |
| Residual constraints | $\sum e_i = 0$, $\sum x_i e_i = 0$ |
| Gaussian noise | MLE = least squares |
| $\sigma^2$ MLE | $\frac{1}{n}\sum e_i^2$ (biased); unbiased: $\frac{1}{n-2}\sum e_i^2$ |
| Laplace noise | MLE = least absolute deviations (robust) |
| Big lesson | **The noise model decides the loss** |

---

## Self-Check Questions

> [!question]- 1. Why does the LS line always pass through $(\bar x, \bar y)$?
> The equation $\partial J/\partial\beta_0 = 0$ gives $\bar y = \beta_0 + \beta_1 \bar x$.

> [!question]- 2. If every $y_i$ is multiplied by 3, what happens to $\hat\beta_1$ and $\hat\beta_0$?
> $S_{xy}$ triples and $S_{xx}$ is unchanged, so $\hat\beta_1$ triples. $\hat\beta_0 = \bar y - \hat\beta_1\bar x$ also triples.

> [!question]- 3. If every $x_i$ is shifted by $+10$, what happens to $\hat\beta_1$?
> Nothing. $S_{xx}$ and $S_{xy}$ use only deviations $x_i - \bar x$, which do not change. (The intercept does change, though.)

> [!question]- 4. Why does the maximiser of $\ln L$ over $(\beta_0,\beta_1)$ not depend on $\sigma^2$?
> $\sigma^2$ only adds a constant and multiplies $J$ by the positive factor $\frac{1}{2\sigma^2}$, so the argmin of $J$ is unchanged.

> [!question]- 5. Why divide by $n-2$, not $n-1$, for the unbiased $\sigma^2$ estimate?
> Two parameters were estimated, which puts two linear constraints on the residuals, leaving $n-2$ degrees of freedom.

> [!question]- 6. If the noise is truly Gaussian but you fit with LAD, do you get the MLE?
> No. LAD is the MLE only under Laplace noise. Under Gaussian noise, LS is the MLE.

> [!question]- 7. If $S_{xx} = 0$, what goes wrong?
> All $x_i$ are equal, so the slope $S_{xy}/S_{xx}$ is undefined. You cannot learn how $y$ changes with $x$ if $x$ never changes.

---

*Next in the slides: slide 9, "Bias and MSE of the estimators".*
