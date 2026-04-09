export interface Sign {
  word: string;
  category: string;
  difficulty: "easy" | "medium" | "hard";
  description: string;
  video?: string;
}

export interface SignsData {
  signs: Sign[];
  categories: string[];
}

export interface Prediction {
  label: string;
  confidence: number;
}

export interface InferenceResult {
  prediction: string;
  confidence: number;
  top3: Prediction[];
  uncertain: boolean;
}
