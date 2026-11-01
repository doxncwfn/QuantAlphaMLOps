\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-report}[20/11/24 v1.0 HCMUT Report Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Move all options to 'article' class
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage[T5]{fontenc}
\RequirePackage{anyfontsize,amsmath,amssymb}
\RequirePackage{geometry,tikz,graphicx,indentfirst,setspace}
\RequirePackage{booktabs}
\RequirePackage[center]{caption}
\RequirePackage{fancyhdr,hyperref}
\RequirePackage[super,square,sort]{natbib}

\RequirePackage[english]{babel} % Change to '[vietnamese]' to use Vietnamese labels

% Return current date
\newcommand{\Month}{
  \ifcase \the\month
  \or January
  \or February
  \or March
  \or April
  \or May
  \or June
  \or July
  \or August
  \or September
  \or October
  \or November
  \or December
  \fi % End of \ifcase
}
\newcommand{\DateToday}{
  \the\day\ \Month
}

% Custom commands
\newcommand{\@upperuniname}{VIETNAM NATIONAL UNIVERSITY -- HO CHI MINH CITY}
\newcommand{\@uniname}{HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY}
\newcommand{\@deptname}{FACULTY OF COMPUTER SCIENCE AND ENGINEERING}
\newcommand{\@coursename}{Specialized Project \\ Computer Science}
\newcommand{\@reporttype}{Project Proposal}
\renewcommand{\@title}{An End-to-End MLOps Pipeline for Cross-Sectional Equity Ranking under Data Constraints: Engineering Validation and Ablation Study.}
\newcommand{\@advisor}{& Dr. Khuong Nguyen An &\\
  & Dr. Vinh Pham An &}
\newcommand{\@stuname}{& Cuong Doan Phuong Hung & ID 2310381}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}
\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}
\newcommand{\@reporttime}{\the\day \Month \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

% Configurations
\geometry{
  paper=a4paper,
  left=2cm,
  right=2cm,
  top=1.5cm,
  bottom=1.5cm,
  includeheadfoot=true,
  headheight=30pt
}
\onehalfspacing
\raggedbottom

% Spacing settings
\renewcommand{\arraystretch}{1.2}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0.3pt}
\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

\AtBeginDocument{\counterwithin{equation}{section}}
\AtBeginDocument{\counterwithin{table}{section}}
\AtBeginDocument{\counterwithin{figure}{section}}

\AtBeginDocument{\renewcommand*\bibname{References}}

\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=red,
  urlcolor=blue
}

%Make cover page
\newcommand{\coverpage}{
  \begin{tikzpicture}[remember picture,overlay,inner sep=0,outer sep=0]
    \draw[blue!70!black,line width=4pt]
    ([xshift=-1.5cm,yshift=-2cm]current page.north east) coordinate (A)--
    ([xshift=1.5cm,yshift=-2cm]current page.north west) coordinate(B)--
    ([xshift=1.5cm,yshift=2cm]current page.south west) coordinate (C)--
    ([xshift=-1.5cm,yshift=2cm]current page.south east) coordinate(D)--cycle;

    \draw
    ([yshift=0.5cm,xshift=-0.5cm]A)--
    ([yshift=0.5cm,xshift=0.5cm]B)--
    ([yshift=-0.5cm,xshift=0.5cm]B) --
    ([yshift=-0.5cm,xshift=-0.5cm]B)--
    ([yshift=0.5cm,xshift=-0.5cm]C)--
    ([yshift=0.5cm,xshift=0.5cm]C)--
    ([yshift=-0.5cm,xshift=0.5cm]C)--
    ([yshift=-0.5cm,xshift=-0.5cm]D)--
    ([yshift=0.5cm,xshift=-0.5cm]D)--
    ([yshift=0.5cm,xshift=0.5cm]D)--
    ([yshift=-0.5cm,xshift=0.5cm]A)--
    ([yshift=-0.5cm,xshift=-0.5cm]A)--
    ([yshift=0.5cm,xshift=-0.5cm]A);

    \draw
    ([yshift=-0.3cm,xshift=0.3cm]A)--
    ([yshift=-0.3cm,xshift=-0.3cm]B)--
    ([yshift=0.3cm,xshift=-0.3cm]B) --
    ([yshift=0.3cm,xshift=0.3cm]B)--
    ([yshift=-0.3cm,xshift=0.3cm]C)--
    ([yshift=-0.3cm,xshift=-0.3cm]C)--
    ([yshift=0.3cm,xshift=-0.3cm]C)--
    ([yshift=0.3cm,xshift=0.3cm]D)--
    ([yshift=-0.3cm,xshift=0.3cm]D)--
    ([yshift=-0.3cm,xshift=-0.3cm]D)--
    ([yshift=0.3cm,xshift=-0.3cm]A)--
    ([yshift=0.3cm,xshift=0.3cm]A)--
    ([yshift=-0.3cm,xshift=0.3cm]A);
  \end{tikzpicture}

  \pagestyle{empty}
  \vspace*{-3\baselineskip}

  \begin{center}
    \MakeUppercase{\@upperuniname}

    \MakeUppercase{\@uniname}

    \MakeUppercase{\@deptname}
    \vfill
    \includegraphics[scale=0.45]{hcmut-logo.png}
    \vfill
    \bfseries
    \begin{tabular}{c}
      \LARGE\@coursename                                  \\
      {}                                                  \\
      \midrule                                            \\
      \Large\@reporttype                                  \\
      {}                                                  \\
      \Large\parbox[c]{0.9\linewidth}{\centering \@title} \\
      {}                                                  \\
      \midrule
    \end{tabular}
    \vfill
    \mdseries
    \begin{tabular}{rll}
      Instructors: \@advisor \\
      Student: \@stuname     \\
    \end{tabular}
    \vfill

    \MakeUppercase{\@reportplace, \@reporttime}
  \end{center}

  \clearpage
  \pagestyle{fancy}
  \newpage
}

% Header and footer
\pagestyle{fancy}
\fancyhf{}

\if@twoside
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
    \end{tabular}
  }
\fi

\if@twoside
  % Even pages (left side)
  \fancyhead[EL]{
    \bfseries
    \begin{tabular}{rl}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
      \begin{tabular}{l}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
    \end{tabular}
  }

  % Odd pages (right side)
  \fancyhead[OR]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
    \end{tabular}
  }
\else
  \fancyhead[R]{
    \bfseries
    \begin{tabular}{rl}
      \begin{tabular}{r}
        \ttfamily\@uniname  \\
        \ttfamily\@deptname \\
      \end{tabular}
      \begin{picture}(25pt,15pt)
        \put(0,-8pt){\includegraphics[width=20pt]{hcmut-logo.png}}
      \end{picture}
    \end{tabular}
  }
\fi