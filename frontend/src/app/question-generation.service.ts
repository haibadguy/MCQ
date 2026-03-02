import { Injectable } from '@angular/core';
import { Observable, throwError } from 'rxjs';
import { HttpClient, HttpHeaders, HttpErrorResponse } from '@angular/common/http';
import { catchError } from 'rxjs/operators';

import { Question } from './_models/quesiton';
import { questionGenerationRequest } from './_models/questionGenerationRequest';

@Injectable({
  providedIn: 'root'
})
export class QuestionGenerationService {

  private baseUrl = 'http://localhost:9002';

  httpOptions = {
    headers: new HttpHeaders({ 'Content-Type': 'application/json' })
  };

  constructor(private http: HttpClient) { }

  /** Generate MCQs from plain text */
  generate(req: questionGenerationRequest): Observable<any[]> {
    return this.http.post<any[]>(`${this.baseUrl}/generate`, req, this.httpOptions).pipe(
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Lỗi kết nối Backend:', error.message);
        return throwError(error);
      })
    );
  }

  /** Generate MCQs from a PDF file (multipart/form-data) */
  generateFromPdf(file: File, count: number): Observable<any[]> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    formData.append('count', String(count));

    return this.http.post<any[]>(`${this.baseUrl}/generate/pdf`, formData).pipe(
      catchError((error: HttpErrorResponse) => {
        console.error('❌ Lỗi PDF Backend:', error.message);
        return throwError(error);
      })
    );
  }
}
