%======================================================================
% hcmut-report.cls
% HCMUT technical report class (built on top of the article class)
%
% File map:
%   1.  Class identification and base class
%   2.  Package loading
%   3.  Colour definitions
%   4.  Page geometry, spacing and numbering depth
%   5.  Hyperref, cross-references and document hooks
%   6.  Date helper commands
%   7.  Report metadata (customisable with setter macros)
%   8.  \startreport - opening page setup
%   9.  Source-code listing styles and helper commands
%   10. Cover page
%   11. Headers and footers
%======================================================================

%---------------------------------------------------------------------
% 1. Class identification and base class
%---------------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{hcmut-report}[20/11/24 v1.0 HCMUT Report Class]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % All options go to 'article'
\ProcessOptions\relax
\LoadClass[12pt,a4paper]{article}

%---------------------------------------------------------------------
% 2. Package loading
%---------------------------------------------------------------------
% --- Text encoding and fonts ---
\RequirePackage[utf8]{vntex,inputenc}
\RequirePackage[T5]{fontenc}
\RequirePackage{anyfontsize}
\RequirePackage[protrusion=false]{microtype}

% --- Localisation ---
\RequirePackage[english]{babel} % Change to '[vietnamese]' to use Vietnamese labels

% --- Mathematics and colour ---
\RequirePackage{amsmath,amssymb}
\RequirePackage{xcolor}

% --- Page layout and spacing ---
\RequirePackage{geometry}
\RequirePackage{indentfirst,setspace}
\RequirePackage{enumitem}

% --- Graphics and figures ---
\RequirePackage{graphicx,tikz}
\RequirePackage[center]{caption}
\RequirePackage{subcaption,float}
\RequirePackage{pgfplots}

% --- Tables ---
\RequirePackage{booktabs,array,multirow,multicol}
\RequirePackage{longtable,tabularx,makecell}

% --- Code listings ---
\RequirePackage{listings}

% --- Boxes, headers, page references, links and bibliography ---
\RequirePackage{tcolorbox}
\RequirePackage{fancyhdr,lastpage}
\RequirePackage{hyperref}
\RequirePackage[super,square,sort]{natbib}
\RequirePackage{acronym}

%---------------------------------------------------------------------
% 3. Colour definitions
%---------------------------------------------------------------------
\definecolor{keywordcolor}{rgb}{0.13,0.13,1}      % Blue keywords
\definecolor{stringcolor}{rgb}{0.7,0.13,0.13}     % Dark red strings
\definecolor{commentcolor}{rgb}{0.25,0.5,0.35}    % Green comments
\definecolor{backgroundcolor}{rgb}{0.97,0.97,0.97}% Light background
\definecolor{numbercolor}{rgb}{0.85,0.65,0.13}    % Numbers
\definecolor{modal-color}{rgb}{0.10,0.74,0.33}    % Modal logo

%---------------------------------------------------------------------
% 4. Page geometry, spacing and numbering depth
%---------------------------------------------------------------------
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

\renewcommand{\arraystretch}{1.2}
\renewcommand{\headrulewidth}{0.3pt}
\renewcommand{\footrulewidth}{0.3pt}

\setcounter{secnumdepth}{3}
\setcounter{tocdepth}{3}

%---------------------------------------------------------------------
% 5. Hyperref, cross-references and document hooks
%---------------------------------------------------------------------
\hypersetup{
    colorlinks=true,
    linkcolor=black,
    citecolor=red,
    urlcolor=blue
}

\AtBeginDocument{\renewcommand*{\bibname}{References}}

% Number equations, tables, figures and listings per section
\AtBeginDocument{%
    \counterwithin{equation}{section}%
    \counterwithin{table}{section}%
    \counterwithin{figure}{section}%
    \counterwithin{lstlisting}{section}%
}

%---------------------------------------------------------------------
% 6. Date helper commands
%---------------------------------------------------------------------
% Return current date
\newcommand{\Month}{%
    \ifcase\the\month
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
\newcommand{\DateToday}{%
    \the\day\ \Month
}

%---------------------------------------------------------------------
% 7. Report metadata
% Override any of these from the preamble using the matching setters,
% e.g. \stuname{...}, \reportplace{...}, \reporttime{...}.
%---------------------------------------------------------------------
\newcommand{\@upperuniname}{VIETNAM NATIONAL UNIVERSITY -- HO CHI MINH CITY}
\newcommand{\@uniname}{HO CHI MINH CITY UNIVERSITY OF TECHNOLOGY}
\newcommand{\@deptname}{FACULTY OF COMPUTER SCIENCE AND ENGINEERING}
\newcommand{\@coursename}{Specialized Project -- Artificial Intelligence\\HK261-DAGD1-041}
\newcommand{\@reporttype}{Technical Report}
\renewcommand{\@title}{An End-to-End MLOps Pipeline for Cross-Sectional Equity Ranking under Data Constraints: Engineering Validation and Ablation Study.}
\newcommand{\@advisor}{& Dr. Pham An Vinh, Ph.D & ID 004361\\
    & Dr. Nguyen An Khuong, Ph.D & ID 003634}
\newcommand{\@stuname}{& Doan Phuong Hung Cuong & ID 2310381\\
    & Nguyen Thien Hao & ID 2310853\\
    & Pham Quang Hieu& ID 2310974}
\newcommand{\stuname}[1]{\renewcommand{\@stuname}{#1}}
\newcommand{\@reportplace}{Ho Chi Minh City}
\newcommand{\reportplace}[1]{\renewcommand{\@reportplace}{#1}}
\newcommand{\@reporttime}{\the\day\ \Month\ \the\year}
\newcommand{\reporttime}[1]{\renewcommand{\@reporttime}{#1}}

%---------------------------------------------------------------------
% 8. \startreport - opening page setup
%---------------------------------------------------------------------
\newcommand{\startfooter}[1]{%
    \fancyfoot[L]{%
        \scriptsize
        \ttfamily
        #1
    }

    \fancyfoot[R]{%
        \scriptsize
        \ttfamily
        Page \thepage/\pageref{LastPage}%
    }%
}
% Table of Contents
\let\originaltableofcontents\tableofcontents
\renewcommand{\tableofcontents}{%
    \begingroup
    \renewcommand{\contentsname}{\huge\bfseries Contents}%
    \originaltableofcontents
    \endgroup
}

% List of Figures
\let\originallistoffigures\listoffigures
\renewcommand{\listoffigures}{%
    \begingroup
    \renewcommand{\listfigurename}{\huge\bfseries List of Figures}%
    \originallistoffigures
    \endgroup
}

% List of Tables
\let\originallistoftables\listoftables
\renewcommand{\listoftables}{%
    \begingroup
    \renewcommand{\listtablename}{\huge\bfseries List of Tables}%
    \originallistoftables
    \endgroup
}

\newcommand{\inlinelogo}[1]{%
    \includegraphics[height=1em]{#1}%
}
%---------------------------------------------------------------------
% 9. Source-code listing styles and helper commands
%---------------------------------------------------------------------
% Shared global style (override per snippet with lstlisting options).
% Individual languages below define their own keywords/comments/strings.
\lstset{
    basicstyle=\ttfamily\small,
    backgroundcolor=\color{white},
    frame=single,
    numbers=left,
    numberstyle=\tiny\color{gray},
    stepnumber=1,
    numbersep=10pt,
    tabsize=2,
    showspaces=false,
    showstringspaces=false,
    breaklines=true,
    breakatwhitespace=true,
    columns=flexible,
    keepspaces=true,
}

% --- Python ---
\lstdefinelanguage{Python}{
morekeywords={
        and, as, assert, async, await, break, class, continue, def, del, elif, else, except,
        False, finally, for, from, global, if, import, in, is, lambda, None, nonlocal, not,
        or, pass, raise, return, True, try, while, with, yield
    },
keywordstyle=\color{keywordcolor}\bfseries,
morekeywords=[2]{bool, bytes, bytearray, complex, dict, float, frozenset, int, list, object, set, str, tuple},
keywordstyle=[2]\color{blue}\bfseries,
identifierstyle=\color{black},
sensitive=true,
comment=[l]{\#},
morecomment=[s]{'''}{'''},
morecomment=[s]{"""}{"""},
commentstyle=\color{commentcolor}\ttfamily,
stringstyle=\color{stringcolor}\ttfamily,
morestring=[b]',
morestring=[b]",
% Highlight numbers
literate=
    *{0}{{{\color{numbercolor}0}}}1
{1}{{{\color{numbercolor}1}}}1
{2}{{{\color{numbercolor}2}}}1
{3}{{{\color{numbercolor}3}}}1
{4}{{{\color{numbercolor}4}}}1
{5}{{{\color{numbercolor}5}}}1
{6}{{{\color{numbercolor}6}}}1
{7}{{{\color{numbercolor}7}}}1
{8}{{{\color{numbercolor}8}}}1
{9}{{{\color{numbercolor}9}}}1
{.0}{{{\color{numbercolor}.0}}}2
{.1}{{{\color{numbercolor}.1}}}2
{.2}{{{\color{numbercolor}.2}}}2
{.3}{{{\color{numbercolor}.3}}}2
{.4}{{{\color{numbercolor}.4}}}2
{.5}{{{\color{numbercolor}.5}}}2
{.6}{{{\color{numbercolor}.6}}}2
{.7}{{{\color{numbercolor}.7}}}2
{.8}{{{\color{numbercolor}.8}}}2
{.9}{{{\color{numbercolor}.9}}}2
{e+}{{{\color{numbercolor}e+}}}2
{e-}{{{\color{numbercolor}e-}}}2
{\#\#NUMSTART\#\#}{}{0}
{\#\#NUMEND\#\#}{}{0},
}

% --- TypeScript ---
\lstdefinelanguage{TypeScript}{
    keywords={abstract, as, boolean, break, case, catch, class, continue, const, constructor, debugger, declare, default, delete, do, else, enum, export, extends, false, finally, for, from, function, get, if, implements, import, in, infer, instanceof, interface, is, keyof, let, module, namespace, never, new, null, number, object, of, package, private, protected, public, readonly, require, global, return, set, static, string, super, switch, symbol, this, throw, true, try, type, typeof, undefined, unique, unknown, var, void, while, with, yield, async, await},
    keywordstyle=\color{keywordcolor}\bfseries,
    ndkeywords={string, number, boolean, Promise, any, void},
    ndkeywordstyle=\color{blue}\bfseries,
    identifierstyle=\color{black},
    sensitive=true,
    comment=[l]{//},
    morecomment=[s]{/*}{*/},
    commentstyle=\color{commentcolor}\ttfamily,
    stringstyle=\color{stringcolor}\ttfamily,
    morestring=[b]',
    morestring=[b]",
}

% --- Environment-variable files (.env) ---
\lstdefinelanguage{EnvCustom}{
    keywords={VITE_APP_API_URL, DATABASE_URL, DB_HOST, DB_USER, DB_PASSWORD, DB_PORT, DB_NAME, JWT_SECRET},
    keywordstyle=\color{blue}\bfseries,
    sensitive=true,
    comment=[l]{\#},
    commentstyle=\color{gray}\ttfamily,
    morestring=[b]',
    morestring=[b]",
    stringstyle=\color{red}\ttfamily,
    identifierstyle=\color{black}
}

% --- JavaScript ---
\lstdefinelanguage{JavaScriptCustom}{
    keywords={
            break, case, catch, class, const, continue, debugger, default, delete, do, else, export,
            extends, finally, for, function, if, import, in, instanceof, let, new, return, super,
            switch, this, throw, try, typeof, var, void, while, with, yield, await, async,
            mysql, dotenv, process, config, createPool, getConnection, release
        },
    keywordstyle=\color{blue}\bfseries,
    ndkeywords={true, false, null, undefined, NaN, Infinity},
    ndkeywordstyle=\color{teal}\bfseries,
    identifierstyle=\color{black},
    sensitive=true,
    comment=[l]{//},
    morecomment=[s]{/*}{*/},
    commentstyle=\color{gray}\ttfamily,
    stringstyle=\color{red}\ttfamily,
    morestring=[b]',
    morestring=[b]"
}

% --- SQL ---
\lstdefinelanguage{SQL}{
    keywords={SELECT, FROM, WHERE, GROUP BY, HAVING, ORDER, BY, CREATE, PROCEDURE, FUNCTION, TRIGGER, BEGIN, END, DECLARE, IF, ELSE, RETURN, SET, INSERT, UPDATE, DELETE, INTO, VALUES, SIGNAL, JOIN, ON, AND, IN, GROUP, AS, RETURNS, DETERMINISTIC, COUNT, IFNULL, SQL, DATA, READS, THEN, AFTER, FOR, EACH, ROW, OLD, NEW, ROUND, CONSTRAINT, FOREIGN, KEY, REFERENCES, PRIMARY, CHECK, TABLE, DEFAULT, DESC, UNION, ALL, OVER, WHILE, DO, LEAVE, EXISTS, BEFORE, OR, IDENTIFIED, GRANT, TO, LEFT, RIGHT},
    keywordstyle=\color{blue}\bfseries,
    morekeywords=[2]{INT, CHAR, VARCHAR, BOOLEAN, DATE, DECIMAL, ENUM, TIME, CASCADE, NULL, NOT, USER, PRIVILEGES, TRUE},
    keywordstyle=[2]\color{red}\bfseries,
    comment=[l]{--},
    commentstyle=\color{gray}\ttfamily,
    morestring=[b]',
    stringstyle=\color{teal},
    basicstyle=\ttfamily\small,
    showstringspaces=false,
    breaklines=true
}

% Inline helper commands: wrap code in one call
\newcommand{\sql}[1]
{
    \begin{lstlisting}[language=SQL]
    #1
  \end{lstlisting}
}
\newcommand{\python}[1]
{
    \begin{lstlisting}[language=Python]
    #1
  \end{lstlisting}
}

%---------------------------------------------------------------------
% 10. Cover page
%---------------------------------------------------------------------
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
        \includegraphics[scale=0.45]{images/hcmut-logo.png}
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
            Supervisors: \@advisor \\
            Students: \@stuname    \\
        \end{tabular}
        \vfill

        \MakeUppercase{\@reportplace, \@reporttime}
    \end{center}

    \clearpage
    \pagestyle{fancy}
    \newpage
}

%---------------------------------------------------------------------
% 11. Headers and footers
%---------------------------------------------------------------------
\pagestyle{fancy}
\fancyhf{}

\if@twoside
    % Even pages (left side): logo on the left
    \fancyhead[EL]{%
        \bfseries
        \begin{tabular}{c l}
            $\vcenter{\hbox{\includegraphics[width=20pt]{images/hcmut-logo.png}}}$ &
            \begin{tabular}{@{}l@{}}
                \ttfamily\@uniname \\
                \ttfamily\@deptname
            \end{tabular}
        \end{tabular}%
    }

    % Odd pages (right side): logo on the right
    \fancyhead[OR]{%
        \bfseries
        \begin{tabular}{r c}
            \begin{tabular}{@{}r@{}}
                \ttfamily\@uniname \\
                \ttfamily\@deptname
            \end{tabular} &
            $\vcenter{\hbox{\includegraphics[width=20pt]{images/hcmut-logo.png}}}$
        \end{tabular}%
    }
\else
    % Single-sided: logo on the right
    \fancyhead[R]{%
        \bfseries
        \begin{tabular}{r c}
            \begin{tabular}{@{}r@{}}
                \ttfamily\@uniname \\
                \ttfamily\@deptname
            \end{tabular} &
            $\vcenter{\hbox{\includegraphics[width=20pt]{images/hcmut-logo.png}}}$
        \end{tabular}%
    }
\fi