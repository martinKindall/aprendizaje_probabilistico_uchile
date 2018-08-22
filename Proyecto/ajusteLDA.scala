// Stanford TMT Example 6 - Training a LabeledLDA model
// http://nlp.stanford.edu/software/tmt/0.4/

// tells Scala where to find the TMT classes
import scalanlp.io._;
import scalanlp.stage._;
import scalanlp.stage.text._;
import scalanlp.text.tokenize._;
import scalanlp.pipes.Pipes.global._;

import edu.stanford.nlp.tmt.stage._;
import edu.stanford.nlp.tmt.model.lda._;
import edu.stanford.nlp.tmt.model.llda._;

val source = CSVFile("data_limpia_train.csv") ~> Drop(1);

val tokenizer = {
  WhitespaceTokenizer() ~>            
  CaseFolder() ~>                        // lowercase everything
  WordsAndNumbersOnlyFilter() ~>         // ignore non-words and non-numbers
  MinimumLengthFilter(3)                 // take terms with >=3 characters
}

val text = {
  source ~>                              // read from the source file
  Column(2) ~>                           // select column containing text
  TokenizeWith(tokenizer) ~>             // tokenize with tokenizer above
  TermCounter() ~>                       // collect counts (needed below)
  TermStopListFilter(List("ante",
"bajo",
"con",
"sin",
"sino",
"todo",
"el",
"ella",
"ellos",
"ellas",
"contra",
"desde",
"hacia",
"para",
"por",
"segun",
"sobre",
"tras",
"durante",
"mientras",
"entre",
"solamente",
"soy",
"ultimo",
"ultima",
"aunque",
"cabe",
"con",
"que",
"los",
"las",
"del",
"esta",
"asi",
"tambien",
"debe",
"sus",
"mas",
"como",
"este",
"los",
"ahora",
"hay",
"algo",
"una",
"uno",
"unos",
"unas",
"cuando",
"sea",
"esto",
"les",
"pero",
"ser",
"sera",
"nos",
"ppk"
)) ~> 
  TermMinimumDocumentCountFilter(4) ~>   // filter terms in <4 docs
  //TermDynamicStopListFilter(5) ~>       // filter out 30 most common terms
  DocumentMinimumLengthFilter(3)         // take only docs with >=5 terms
}

// define fields from the dataset we are going to slice against
val labels = {
  source ~>                              // read from the source file
  Column(3) ~>                           // take column two, the year
  TokenizeWith(WhitespaceTokenizer()) ~> // turns label field into an array
  TermCounter() ~>                       // collect label counts
  TermMinimumDocumentCountFilter(10)     // filter labels in < 10 docs
}

val dataset = LabeledLDADataset(text, labels);

// define the model parameters
val modelParams = LabeledLDAModelParams(dataset);

// Name of the output model folder to generate
val modelPath = file("training_model");

// Trains the model, writing to the given output path
TrainCVB0LabeledLDA(modelParams, dataset, output = modelPath, maxIterations = 1000);
// or could use TrainGibbsLabeledLDA(modelParams, dataset, output = modelPath, maxIterations = 1500);

