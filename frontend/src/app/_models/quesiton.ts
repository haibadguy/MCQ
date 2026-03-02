export class Question {
  answerText: string;
  questionText: string;
  distractors: string[];
  /** ISO-639-1 language code returned by the backend pipeline detector.
   *  'en' → English T5 pipeline | 'vi' → Vietnamese ViT5 pipeline */
  lang: string = 'en';
  /** UI-only: shuffled array of [distractors + answerText] for rendering */
  answers: string[] = [];
}